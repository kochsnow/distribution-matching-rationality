import os
import time
import sys
sys.path.append("./core")
import argparse
import numpy as np
from misc import str2bool
import pdb
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.enable_eager_execution(config=config)

from beer import get_beer_dataset, get_beer_annotation
from language import get_pretained_glove
from model import TargetRNN, TeacherEncoder
from train_utils import train, train_teacher
from eval_utils import flush, validate

# set random seed
tf.set_random_seed(22252018)
np.random.seed(22252018)

parser = argparse.ArgumentParser(
    description="classwise rationalization for beer review.")

# dataset parameters
parser.add_argument('--data_dir',
                    type=str,
                    required=True,
                    help='Path of the dataset')
parser.add_argument(
    '--balance',
    type=str2bool,
    default=False,
    help='Balance the data for each class or not [default: False]')
parser.add_argument('--aspect',
                    type=int,
                    required=True,
                    help='The aspect number of beer review [0, 1, 2]')
parser.add_argument('--annotation_path',
                    type=str,
                    default=None,
                    help='Path to the annotation')
parser.add_argument('--max_seq_length',
                    type=int,
                    default=256,
                    help='Max sequence length [default: 256]')
parser.add_argument('--word_threshold',
                    type=int,
                    default=2,
                    help='Min frequency to keep a word [default: 2]')
parser.add_argument('--batch_size',
                    type=int,
                    default=100,
                    help='Batch size [default: 100]')
parser.add_argument('--shuffle_buffer_size',
                    type=int,
                    default=100000,
                    help='Buffer size for data shuffling [default: 100000]')

# pretrained embeddings
parser.add_argument('--embedding_dir',
                    type=str,
                    default=None,
                    help='Dir. of pretrained embeddings [default: None]')
parser.add_argument('--embedding_name',
                    type=str,
                    default=None,
                    help='File name of pretrained embeddings [default: None]')

# model parameters
parser.add_argument('--cell_type',
                    type=str,
                    default="GRU",
                    help='Cell type: LSTM, GRU [default: GRU]')
parser.add_argument('--embedding_dim',
                    type=int,
                    default=100,
                    help='Embedding dims [default: 100]')
parser.add_argument('--hidden_dim',
                    type=int,
                    default=100,
                    help='RNN hidden dims [default: 100]')
parser.add_argument('--num_classes',
                    type=int,
                    default=2,
                    help='Number of predicted classes [default: 2]')

# ckpt parameters
parser.add_argument('--output_dir',
                    type=str,
                    required=True,
                    help='Base dir of output files')

# learning parameters
parser.add_argument('--num_epchos',
                    type=int,
                    required=True,
                    help='Number of training epoch')
parser.add_argument('--gen_pos_lr',
                    type=float,
                    default=1e-3,
                    help='Positive generator learning rate [default: 1e-3]')
parser.add_argument('--gen_neg_lr',
                    type=float,
                    default=1e-3,
                    help='Negative generator  learning rate [default: 1e-3]')
parser.add_argument('--discriminator_lr',
                    type=float,
                    default=1e-3,
                    help='Discriminator learning rate [default: 1e-3]')
parser.add_argument('--sparsity_lambda',
                    type=float,
                    default=1.,
                    help='Sparsity trade-off [default: 1.]')
parser.add_argument('--continuity_lambda',
                    type=float,
                    default=4.,
                    help='Continuity trade-off [default: 4.]')
parser.add_argument(
    '--sparsity_percentage',
    type=float,
    default=0.2,
    help='Regularizer to control highlight percentage [default: .2]')

# visual parameters
parser.add_argument(
    '--visual_interval',
    type=int,
    default=50,
    help='How frequent to generate a sample of rationale [default: 50]')

# gpu support
parser.add_argument('--gpu',
                    type=str,
                    default=None,
                    help='id(s) for CUDA_VISIBLE_DEVICES [default: None]')

args = parser.parse_args()

print("\nParameters:")
for attr, value in sorted(args.__dict__.items()):
    print("\t{}={}".format(attr.upper(), value))

######################
# set visiable gpu
######################
if args.gpu != None:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

######################
# check output dir
######################
if not tf.gfile.Exists(args.output_dir):
    tf.gfile.MakeDirs(args.output_dir)

######################
# load dataset
######################
train_dataset, dev_dataset, language_index = get_beer_dataset(
    args.data_dir,
    args.max_seq_length,
    args.word_threshold,
    balance=args.balance)

annotation_dataset = get_beer_annotation(args.annotation_path, args.aspect,
                                         args.max_seq_length,
                                         language_index.word2idx)

# shuffle and batch the dataset
train_dataset = train_dataset.shuffle(args.shuffle_buffer_size).batch(
    args.batch_size, drop_remainder=False)
dev_dataset = dev_dataset.batch(args.batch_size, drop_remainder=False)
annotation_dataset = annotation_dataset.batch(args.batch_size,
                                              drop_remainder=False)

######################
# update arguments
######################
args.vocab_size = len(language_index.word2idx)
args.idx2word = language_index.idx2word

if (args.embedding_dir and args.embedding_name):
    # get pretrained embedding
    fembedding = os.path.join(args.embedding_dir, args.embedding_name)
    args.pretrained_embedding = get_pretained_glove(language_index.word2idx,
                                                    fembedding)
else:
    args.pretrained_embedding = None

######################
# define the model and manually build the model
######################
target_rnn = TargetRNN(args)
fake_data = tf.zeros([args.batch_size, args.max_seq_length])
fake_label = tf.zeros([args.batch_size, args.num_classes])
# mannually build model with dummy tensorts
_, _, _ = target_rnn(fake_data, fake_data, fake_label, path=0)
_, _, _ = target_rnn(fake_data, fake_data, fake_label, path=1)

teacher_encoder=TeacherEncoder(args)
true_data = tf.zeros([args.batch_size, args.max_seq_length])
true_label = tf.zeros([args.batch_size, args.num_classes])
_, _ = teacher_encoder.call(true_data, true_data, true_label, path=0)

######################
# Training
######################
gen_pos_optimizer = tf.train.AdamOptimizer(learning_rate=args.gen_pos_lr)
gen_neg_optimizer = tf.train.AdamOptimizer(learning_rate=args.gen_neg_lr)
dis_optimizer = tf.train.AdamOptimizer(learning_rate=args.discriminator_lr)
dis_teacher_optimizer = tf.train.AdamOptimizer(learning_rate=args.discriminator_lr)

gen_pos_step_counter = tf.Variable(0,
                                   trainable=False,
                                   name='gen_pos_step',
                                   dtype=tf.int64)
gen_neg_step_counter = tf.Variable(0,
                                   trainable=False,
                                   name='gen_neg_step',
                                   dtype=tf.int64)
dis_step_counter = tf.Variable(0,
                               trainable=False,
                               name='dis_step',
                               dtype=tf.int64)
dis_teacher_step_counter = tf.Variable(0,
                               trainable=False,
                               name='dis_step',
                               dtype=tf.int64)

optimizers = [gen_pos_optimizer, gen_neg_optimizer, dis_optimizer, dis_teacher_optimizer]
step_counters = [gen_pos_step_counter, gen_neg_step_counter, dis_step_counter, dis_teacher_step_counter]

#training dis-teacher
# 指定检查点目录
# checkpoint_directory = 'checkpoints/teacher_encoder/'
# # 创建模型检查点
# checkpoint = tfe.Checkpoint(optimizer=optimizer,
#                             model=model,
#                             optimizer_step=tf.train.get_or_create_global_step())

for epcho in range(10):
    start = time.time()
    recall, precision, f1_score=train_teacher(teacher_encoder, [dis_teacher_optimizer], train_dataset, [dis_teacher_step_counter], args)
    end = time.time()
    print("traning epoch:{} recall:{:.4f} precision:{:.4f} f1-score:{:.4f}".format(epcho, recall, precision, f1_score))
    print('\nTrain time for epoch #%d (%d total disc steps): %f second' %
          (epcho + 1, dis_step_counter.numpy(), end - start))
    GT=[]
    PR=[]
    for (batch, (inputs, masks, labels)) in enumerate(dev_dataset):
        dis_teacher_logits, _ = teacher_encoder(inputs, masks, labels, path=0)
        #pdb.set_trace()
        predicted = tf.argmax(dis_teacher_logits, axis=1, output_type=tf.int32)
        actual = tf.argmax(labels, axis=1, output_type=tf.int32)
        PR.append(predicted)
        GT.append(actual)
        # compute accuarcy
    PR=tf.concat(PR, 0)
    GT=tf.concat(GT,0)
    TP = tf.count_nonzero(PR * GT)
    TN = tf.count_nonzero((PR - 1) * (GT - 1))
    FP = tf.count_nonzero(PR * (GT - 1))
    FN = tf.count_nonzero((PR - 1) * GT)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2*precision*recall/(recall+precision)
    print("dev epoch:{} recall:{:.4f} precision:{:.4f} f1-score:{:.4f}".format(epcho, recall, precision, f1_score))

#  training
for epcho in range(args.num_epchos):
    start = time.time()
    train(target_rnn, teacher_encoder, optimizers, train_dataset, step_counters, args)
    end = time.time()
    print('\nTrain time for epoch #%d (%d total disc steps): %f second' %
          (epcho + 1, dis_step_counter.numpy(), end - start))

# validation
    print("Validate with huamn annotations")
    annotation_results = validate(target_rnn,
                                  annotation_dataset,
                                  args.idx2word,
                                  visual_interval=args.visual_interval,
                                  file=os.path.join(args.output_dir,
                                                    "visual_ann.txt"))

    print(
        "The annotation performance: sparsity: %.4f, precision: %.4f, recall: %.4f, f1: %.4f"
        % (100 * annotation_results[0], 100 * annotation_results[1],
           100 * annotation_results[2], 100 * annotation_results[3]))

# output the results
flush(target_rnn, dev_dataset, args.idx2word,
      os.path.join(args.output_dir, "visual_dev.txt"))
