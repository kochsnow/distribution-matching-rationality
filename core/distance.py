import tensorflow as tf
def l2diff(x1, x2):
    """
    standard euclidean norms
    """
    return tf.math.sqrt(tf.cast(tf.reduce_sum((x1 - x2) ** 2), "float32"))


def moment_diff(sx1, sx2, k):
    """
    difference between moments
    """
    #print(sx1, sx2)
    ss1 = tf.reduce_mean(sx1 * tf.cast(k, 'float32'), 0)
    ss2 = tf.reduce_mean(sx2 * tf.cast(k, 'float32'), 0)
    return l2diff(ss1, ss2)


def gaussian_kernel(x1, x2, beta=1.0):
    r = x1.dimshuffle(0, 'x', 1)
    return tf.exp(-beta * tf.square(r - x2).sum(axis=-1))


class Distance(object):
    def __init__(self, distance_name="cmd", beta=1.0):
        if distance_name == "cmd":
            self.distance = self.cmd
        elif distance_name == "mmd":
            self.distance = self.mmd
            self.beta = beta
        elif distance_name == "coral":
            self.coral = self.mmd

    def cmd(self, x1, x2, n_moments=5):
        mx1 = tf.reduce_mean(x1, 0)
        mx2 = tf.reduce_mean(x2, 0)
        sx1 = x1 - mx1
        sx2 = x2 - mx2
        dm = l2diff(mx1, mx2)
        for i in range(n_moments - 1):
            dm += moment_diff(sx1, sx2, i + 2)
        return dm


    def mmd(self, x1, x2):
        x1x1 = gaussian_kernel(x1, x1, beta=self.beta)
        x1x2 = gaussian_kernel(x1, x2, beta=self.beta)
        x2x2 = gaussian_kernel(x2, x2, beta=self.beta)
        diff = x1x1.mean() - 2 * x1x2.mean() + x2x2.mean()
        return diff

    def coral(self, x1, x2):
        c1 = 1. / (x1.shape[0] - 1) * (tf.tensordot(tf.transpose(x1), x1) -
                                       tf.tensordot(tf.transpose(x1.mean(axis=0)), x1.sum(axis=0)))
        c2 = 1. / (x2.shape[0] - 1) * (tf.tensordot(tf.transpose(x2), x2) -
                                       tf.tensordot(tf.transpose(x2.mean(axis=0)), x2.sum(axis=0)))
        return 1. / (4 * x1.shape[0] ** 2) * ((c1 - c2) ** 2).sum()