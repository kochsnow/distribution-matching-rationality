import tensorflow as tf
from tensorflow.keras.backend import dot
tf.enable_eager_execution()


def l2diff(x1, x2):
    """
    standard euclidean norm
    """
    return tf.math.sqrt(tf.cast(tf.reduce_sum((x1 - x2) ** 2), "float32"))


def moment_diff(sx1, sx2, k):
    """
    difference between moments
    """
    ss1 = tf.reduce_mean(sx1 * tf.cast(k, 'float32'), 0)
    ss2 = tf.reduce_mean(sx2 * tf.cast(k, 'float32'), 0)
    return l2diff(ss1, ss2)


def gaussian_kernel(x1, x2, beta=1.0):
    r = tf.transpose(x1, [0, 1], 0)
    return tf.reduce_sum(tf.exp(-beta * tf.square(r - x2)), axis=-1)


class Distance(object):
    def __init__(self, distance_name="cmd", beta=1.0):
        if distance_name == "cmd":
            self.distance = self.cmd
        elif distance_name == "mmd":
            self.distance = self.mmd
            self.beta = beta
        elif distance_name == "coral":
            self.distance = self.coral

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
        diff = tf.reduce_mean(x1x1) - 2 * tf.reduce_mean(x1x2) + tf.reduce_mean(x2x2)
        return diff

    def coral(self, x1, x2):
        print(x1.shape[0] - 1)
        c1 = 1. / (x1.shape[0].value - 1) * (dot(tf.transpose(x1), x1)
                                                         - tf.tensordot(tf.transpose(tf.reduce_mean(x1, axis=0)), tf.reduce_sum(x1, axis=0), 0))
        c2 = 1. / (x2.shape[0].value - 1) * (dot(tf.transpose(x2), x2) -
                                       tf.tensordot(tf.transpose(tf.reduce_mean(x2, axis=0)), tf.reduce_sum(x2,axis=0), 0))
        return 1. / tf.reduce_sum((4 * x1.shape[0].value ** 2) * ((c1 - c2) ** 2))

if __name__ == '__main__':
    dist = Distance("cmd")
    a = tf.constant([[4.,2,3,4], [2,2,1,4]])
    b = tf.constant([[2.,2,3,4], [2,2,3,4]])
    print(dist.distance(a, b))

