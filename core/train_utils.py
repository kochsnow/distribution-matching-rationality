import sys
import random
import tensorflow as tf
from metric import get_sparsity_loss, get_continuity_loss, compute_accuracy
from visualize import show_binary_rationale
from distance import Distance
import pdb
#from dist import Distance
def gen_nl_loss(logits, targets, truth, path):
    """
    This is a negative likelihood loss for the generator. 
    In other words, generator max the prob of the targets. 
    Note that, likelihood does not take log.  
    Inputs: 
        logits -- the logits from the disc. (batch_size, num_classes)
        targets -- either all zeros or all ones, 
                   depends on which path it go through.
        truth -- the real label of each examples,
                  this is not used to calculate the loss. 
        path -- whether the rationale is generated by G0 or G1.        
    """
    softmax_probs = tf.nn.softmax(logits=logits, axis=-1)
    probs = tf.reduce_sum(softmax_probs * tf.cast(targets, tf.float32), axis=-1)
    total_loss = -tf.reduce_mean(probs)

    return total_loss

def train_teacher(model, optimizers, dataset, step_counters, args):
    """
    Training target dependent rationale generation 
    (Tommi's three player version).
    """
    #gen_optimizer = optimizers[0]
    dis_teacher_optimizer = optimizers[0]
    #dis_student_optimizer = optimizers[2]

    #gen_counter = step_counters[0]
    dis_teacher_step_counter = step_counters[0]
    #dis_student_counter = step_counters[2]
    path=0
    GT=[]
    PR=[]
    for (batch, (inputs, masks, labels)) in enumerate(dataset):
        # get variables
        #gen_vars = model.generator_trainable_variables()
        dis_teacher_vars = model.discriminator_teacher_trainable_variables()
        #dis_student_vars = model.discriminator_student_trainable_variables()
        #print(dis_student_vars)
        # construct the target labels for the generator
        batch_size = inputs.shape[0]
        all_ones = tf.ones([batch_size, 1], tf.int32)
        all_zeros = tf.zeros(all_ones.shape, tf.int32)

        with tf.GradientTape() as dis_tape:
            dis_teacher_logits, _ = model(inputs, masks, labels, path)
            dis_hard_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=dis_teacher_logits, labels=labels))
            dis_loss = dis_hard_loss
        dis_grads = dis_tape.gradient(dis_loss, dis_teacher_vars)
        dis_teacher_optimizer.apply_gradients(zip(dis_grads, dis_teacher_vars), global_step=dis_teacher_step_counter)
        #pdb.set_trace()
        predicted = tf.argmax(dis_teacher_logits, axis=1, output_type=tf.int32)
        actual = tf.argmax(labels, axis=1, output_type=tf.int32)
        PR.append(predicted)
        GT.append(actual)
        # compute accuarcy
        #dis_accuracy = compute_accuracy(dis_teacher_logits, labels)
        TP = tf.count_nonzero(predicted * actual)
        TN = tf.count_nonzero((predicted - 1) * (actual - 1))
        FP = tf.count_nonzero(predicted * (actual - 1))
        FN = tf.count_nonzero((predicted - 1) * actual)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1_score = 2*precision*recall/(recall+precision)
        if batch%10==0:
            print("{}/{}: recall:{:.4f} precision:{:.4f} f1-score:{:.4f}".format(batch, len(list(dataset)), recall, precision, f1_score))
        if (args.visual_interval and
            ((batch + 1) % args.visual_interval in [0, 1]) and batch != 0):

            batch_size = inputs.get_shape()[0]
            # random a sample from the batch_size
            sample_id = random.randint(0, batch_size - 1)

            print('Batch # %d: sample id: %d, true label: %d, path: %d' %
                  (batch + 1, sample_id, tf.argmax(labels[sample_id, :]), path))
            print("dis acc:{}".format(dis_accuracy))
            show_binary_rationale(inputs[sample_id, :].numpy(),
                                  rationales[sample_id, :, 1].numpy(),
                                  args.idx2word)
            sys.stdout.flush()
    #pdb.set_trace()
    PR=tf.concat(PR, 0)
    GT=tf.concat(GT,0)
    TP = tf.count_nonzero(PR * GT)
    TN = tf.count_nonzero((PR - 1) * (GT - 1))
    FP = tf.count_nonzero(PR * (GT - 1))
    FN = tf.count_nonzero((PR - 1) * GT)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2*precision*recall/(recall+precision)
    return recall, precision, f1_score

def train(model, teacher_encoder, optimizers, dataset, step_counters, args):
    """
    Training target dependent rationale generation 
    (Tommi's three player version).
    """
    print("cls_lambda:{} om_lambda:{} fm_lambda:{}".format(args.cls_lambda, args.om_lambda, args.fm_lambda))
    gen_pos_optimizer = optimizers[0]
    gen_neg_optimizer = optimizers[1]
    dis_optimizer = optimizers[2]

    gen_pos_step_counter = step_counters[0]
    gen_neg_step_counter = step_counters[1]
    dis_step_counter = step_counters[2]
    params_t =args.params_t
    CMD = Distance(args.distance)
    for (batch, (inputs, masks, labels)) in enumerate(dataset):
        # get variables
        gen_pos_vars = model.generator_pos_trainable_variables()
        gen_neg_vars = model.generator_neg_trainable_variables()
        dis_vars = model.discriminator_trainable_variables()
#         print(dis_vars)
        #pdb.set_trace()
        # construct the target labels for the generator
        batch_size = inputs.shape[0]
        all_ones = tf.ones([batch_size, 1], tf.int32)
        all_zeros = tf.zeros(all_ones.shape, tf.int32)

        path = dis_step_counter.numpy() % 2

        if path == 0:
            # go through the G0
            gen_targets = tf.concat([all_ones, all_zeros], axis=-1)
        else:
            # go through the G1
            gen_targets = tf.concat([all_zeros, all_ones], axis=-1)

        with tf.GradientTape() as dis_tape:
            # logits -- (batch_size, num_classes)
            # rationales -- (batch_size, seq_length, 2)
            dis_student_logits, rationales, dis_student_output, dis_student_outputs,\
            dis_student_raw_output, dis_student_raw_outputs = model(inputs, masks, labels, path)
            dis_teacher_logits, dis_teacher_output = teacher_encoder(inputs, masks, labels, path)
            dis_hard_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=dis_student_logits, labels=labels))
            soft_labels = tf.nn.softmax(dis_teacher_logits/params_t)
            dis_soft_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=dis_student_logits, labels=soft_labels))
            #fm loss
            cmd_loss_1 = CMD.distance(dis_student_output, dis_student_raw_output)
            cmd_loss_2 = CMD.distance(dis_student_outputs, dis_student_raw_outputs)
            dis_fm_loss = cmd_loss_1
            #pdb.set_trace()
            dis_loss = args.cls_lambda*dis_hard_loss+args.om_lambda*dis_soft_loss+ args.fm_lambda*dis_fm_loss
            
            if batch%20==0:
                print("{}/{} dis_hard_loss:{:.4f} dis_soft_loss:{:.4f} dis_fm_loss:{:.4f} \
                      cmd_loss_1:{:.4f} cmd_loss_2:{:.4f}".format(
                    batch, len(list(dataset)), dis_hard_loss, dis_soft_loss, dis_fm_loss,
                cmd_loss_1, cmd_loss_2))
            
        # compute graident for the disc
        dis_grads = dis_tape.gradient(dis_loss, dis_vars)

        with tf.GradientTape() as gen_tape:
            dis_logits, rationales, _, _, _, _ = model(inputs, masks, labels, path)

            # generator loss
            class_loss = gen_nl_loss(dis_logits, gen_targets, labels, path)
            sparsity_loss = args.sparsity_lambda * get_sparsity_loss(
                rationales[:, :, 1], tf.cast(masks, tf.float32),
                args.sparsity_percentage)
            continuity_loss = args.continuity_lambda * get_continuity_loss(
                rationales[:, :, 1])
            gen_loss = class_loss + sparsity_loss + continuity_loss

        if path == 0:
            gen_neg_grads = gen_tape.gradient(gen_loss, gen_neg_vars)
            gen_neg_optimizer.apply_gradients(zip(gen_neg_grads, gen_neg_vars),
                                              global_step=gen_neg_step_counter)
        else:
            gen_pos_grads = gen_tape.gradient(gen_loss, gen_pos_vars)
            gen_pos_optimizer.apply_gradients(zip(gen_pos_grads, gen_pos_vars),
                                              global_step=gen_pos_step_counter)

        dis_optimizer.apply_gradients(zip(dis_grads, dis_vars),
                                      global_step=dis_step_counter)

        # compute accuarcy
        dis_accuracy = compute_accuracy(dis_logits, labels)

        if (args.visual_interval and
            ((batch + 1) % args.visual_interval in [0, 1]) and batch != 0):

            batch_size = inputs.get_shape()[0]
            # random a sample from the batch_size
            sample_id = random.randint(0, batch_size - 1)

            print('Batch # %d: sample id: %d, true label: %d, path: %d' %
                  (batch + 1, sample_id, tf.argmax(labels[sample_id, :]), path))
            show_binary_rationale(inputs[sample_id, :].numpy(),
                                  rationales[sample_id, :, 1].numpy(),
                                  args.idx2word)
            sys.stdout.flush()