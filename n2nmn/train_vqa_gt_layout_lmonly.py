from __future__ import absolute_import, division, print_function

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=0)
args = parser.parse_args()

gpu_id = args.gpu_id  # set GPU id to use
import os; os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

import numpy as np
import tensorflow as tf
import sys

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Variable

# Start the session BEFORE importing tensorflow_fold
# to avoid taking up all GPU memory
sess = tf.Session(config=tf.ConfigProto(
    gpu_options=tf.GPUOptions(allow_growth=True),
    allow_soft_placement=False, log_device_placement=False))

from models_vqa.nmn3_assembler import Assembler
from models_vqa.nmn3_model import NMN3Model
from util.vqa_train.data_reader import DataReader

# Module parameters
H_feat = 14
W_feat = 14
D_feat = 2048
embed_dim_txt = 300
embed_dim_nmn = 300
lstm_dim = 1000
num_layers = 2
encoder_dropout = True
decoder_dropout = True
decoder_sampling = False
T_encoder = 26
T_decoder = 13
N = 64
use_qpn = True
qpn_dropout = True
reduce_visfeat_dim = False
glove_mat_file = './exp_vqa/data/vocabulary_vqa_glove.npy'

# Training parameters
weight_decay = 0
baseline_decay = 0.99
max_iter = 40000
snapshot_interval = 5000
exp_name = "vqa_gt_layout"
snapshot_dir = '/dataorig/VQA/exp_vqa/tfmodel/%s/' % exp_name

# Log params
log_interval = 20
log_dir = '/dataorig/VQA/exp_vqa/tb/%s/' % exp_name

# Data files
vocab_question_file = './exp_vqa/data/vocabulary_vqa.txt'
vocab_layout_file = './exp_vqa/data/vocabulary_layout.txt'
vocab_answer_file = './exp_vqa/data/answers_vqa.txt'

imdb_file_trn = './exp_vqa/data/imdb/imdb_trainval2014.npy'

assembler = Assembler(vocab_layout_file)

data_reader_trn = DataReader(imdb_file_trn, shuffle=True, one_pass=False,
                             batch_size=N,
                             T_encoder=T_encoder,
                             T_decoder=T_decoder,
                             assembler=assembler,
                             vocab_question_file=vocab_question_file,
                             vocab_answer_file=vocab_answer_file)

num_vocab_txt = data_reader_trn.batch_loader.vocab_dict.num_vocab
num_vocab_nmn = len(assembler.module_names)
num_choices = data_reader_trn.batch_loader.answer_dict.num_vocab

# Network inputs
input_seq_batch = tf.placeholder(tf.int32, [None, None])
seq_length_batch = tf.placeholder(tf.int32, [None])
image_feat_batch = tf.placeholder(tf.float32, [None, H_feat, W_feat, D_feat])
expr_validity_batch = tf.placeholder(tf.bool, [None])
answer_label_batch = tf.placeholder(tf.int32, [None])
use_gt_layout = tf.constant(False, dtype=tf.bool)
gt_layout_batch = tf.placeholder(tf.int32, [None, None])

# The model for training
nmn3_model_trn = NMN3Model(
    image_feat_batch, input_seq_batch,
    seq_length_batch, T_decoder=T_decoder,
    num_vocab_txt=num_vocab_txt, embed_dim_txt=embed_dim_txt,
    num_vocab_nmn=num_vocab_nmn, embed_dim_nmn=embed_dim_nmn,
    lstm_dim=lstm_dim, num_layers=num_layers,
    assembler=assembler,
    encoder_dropout=encoder_dropout,
    decoder_dropout=decoder_dropout,
    decoder_sampling=decoder_sampling,
    num_choices=num_choices,
    use_qpn=use_qpn, qpn_dropout=qpn_dropout, reduce_visfeat_dim=reduce_visfeat_dim,
    use_gt_layout=use_gt_layout,
    gt_layout_batch=gt_layout_batch)

# Loss function
softmax_loss_per_sample = tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=nmn3_model_trn.scores, labels=answer_label_batch)
# The final per-sample loss, which is vqa loss for valid expr
# and invalid_expr_loss for invalid expr
final_loss_per_sample = softmax_loss_per_sample  # All exprs are valid

avg_sample_loss = tf.reduce_mean(final_loss_per_sample)
seq_likelihood_loss = tf.reduce_mean(-nmn3_model_trn.log_seq_prob)

total_training_loss = seq_likelihood_loss + avg_sample_loss
total_loss = total_training_loss + weight_decay * nmn3_model_trn.l2_reg

# Train with Adam
solver = tf.train.AdamOptimizer()
gradients = solver.compute_gradients(total_loss)

# no gradient clipping
# Clip gradient by L2 norm
# gradients = gradients_part1+gradients_part2
# gradients = [(tf.clip_by_norm(g, max_grad_l2_norm), v)
#              for g, v in gradients]
solver_op = solver.apply_gradients(gradients)

# Training operation
# Partial-run can't fetch training operations
# some workaround to make partial-run work
with tf.control_dependencies([solver_op]):
    train_step = tf.constant(0)

# Write summary to TensorBoard
os.makedirs(log_dir, exist_ok=True)
log_writer = tf.summary.FileWriter(log_dir, tf.get_default_graph())

loss_ph = tf.placeholder(tf.float32, [])
entropy_ph = tf.placeholder(tf.float32, [])
accuracy_ph = tf.placeholder(tf.float32, [])
baseline_ph = tf.placeholder(tf.float32, [])
validity_ph = tf.placeholder(tf.float32, [])
summary_trn = []
summary_trn.append(tf.summary.scalar("avg_sample_loss", loss_ph))
summary_trn.append(tf.summary.scalar("entropy", entropy_ph))
summary_trn.append(tf.summary.scalar("avg_accuracy", accuracy_ph))
# summary_trn.append(tf.summary.scalar("baseline", baseline_ph))
summary_trn.append(tf.summary.scalar("validity", validity_ph))
log_step_trn = tf.summary.merge(summary_trn)

tst_answer_accuracy_ph = tf.placeholder(tf.float32, [])
tst_layout_accuracy_ph = tf.placeholder(tf.float32, [])
tst_layout_validity_ph = tf.placeholder(tf.float32, [])
summary_tst = []
summary_tst.append(tf.summary.scalar("test_answer_accuracy", tst_answer_accuracy_ph))
summary_tst.append(tf.summary.scalar("test_layout_accuracy", tst_layout_accuracy_ph))
summary_tst.append(tf.summary.scalar("test_layout_validity", tst_layout_validity_ph))
log_step_tst = tf.summary.merge(summary_tst)

os.makedirs(snapshot_dir, exist_ok=True)
snapshot_saver = tf.train.Saver(max_to_keep=None)  # keep all snapshots
sess.run(tf.global_variables_initializer())

# Load glove vector
glove_mat = np.load(glove_mat_file)
with tf.variable_scope('neural_module_network/layout_generation/encoder_decoder/encoder', reuse=True):
    embedding_mat = tf.get_variable('embedding_mat')
    sess.run(tf.assign(embedding_mat, glove_mat))


qnword2id = {}
f = open(vocab_question_file)
ctr = 0
for line in f:
   line = line.split('\n')[0]
   if line in qnword2id:
      continue
   else:
      qnword2id[line] = ctr
      ctr += 1

#print(qnword2id)


def get_ids(arr):
   arr_id = []
   for i, a in enumerate(arr):
      #print(i,a)
      words = a.replace(',',' ').replace('.',' ').replace('?',' ').replace("'"," ").replace('-',' ').replace('/', ' / ').replace('>',' > ').replace(':',' : ').split()
      try:
        arr_id.append([qnword2id[k.lower()] for k in words])
      except KeyError:
        arr_id.append([0])
    
   return np.array(arr_id)

class Net(torch.nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.embedding = nn.Embedding(17742, 128)
        self.rnn = nn.LSTM(128, 3001, batch_first=True) # input_dim, hidden_dim
        self.fc1 = nn.Linear(64, 32) # https://stackoverflow.com/questions/45022734/understanding-a-simple-lstm-pytorch
        self.fc2 = nn.Linear(32,3)

    def forward(self,x):

        #print("Shape input to RNN is ", x.shape)
        x  = self.embedding(x) # (batch_size, seq, input_size)
        out, (h,x) = self.rnn(x)
        return h


net = Net()
if torch.cuda.is_available():
   net.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.001)


def run_training(max_iter, dataset_trn):
    avg_accuracy = 0
    accuracy_decay = 0.99
    loss_val = 0
    for n_iter, batch in enumerate(dataset_trn.batches()):

        
        #print("I got batch as ", batch)
        #print("Question strings are: ", batch['qstr_list'], " of length ", len(batch['qstr_list']))
        #print("Answers are: ", batch['answer_label_batch'], " of length ", len(batch['answer_label_batch']))
        #print("GT Layouts are: ", batch['gt_layout_batch'], " of length ", len(batch['gt_layout_batch'][0]))
        #print("Image Features are of length ", len(batch['image_feat_batch']), batch['image_feat_batch'][0].shape)
        #print("Sequences Lengths are of length: ", len(batch['seq_length_batch']))
         

        if n_iter >= max_iter:
            break
        # set up input and output tensors
        h = sess.partial_run_setup(
            [nmn3_model_trn.predicted_tokens, nmn3_model_trn.entropy_reg,
             nmn3_model_trn.scores, avg_sample_loss, train_step],
            [input_seq_batch, seq_length_batch, image_feat_batch,
             nmn3_model_trn.compiler.loom_input_tensor, expr_validity_batch,
             answer_label_batch, gt_layout_batch])

        # Part 0 & 1: Run Convnet and generate module layout
        tokens, entropy_reg_val = sess.partial_run(h,
            (nmn3_model_trn.predicted_tokens, nmn3_model_trn.entropy_reg),
            feed_dict={input_seq_batch: batch['input_seq_batch'],
                       seq_length_batch: batch['seq_length_batch'],
                       image_feat_batch: batch['image_feat_batch'],
                       gt_layout_batch: batch['gt_layout_batch']})
        # Assemble the layout tokens into network structure
        expr_list, expr_validity_array = assembler.assemble(tokens)
        # all exprs should be valid (since they are ground-truth)
        assert(np.all(expr_validity_array))

        labels = batch['answer_label_batch']
        questions = batch['qstr_list']
        questions_id = get_ids(questions)
        #print("Shape of questions_id: ", questions_id.shape, " and that of labels: ", labels.shape)

        scores_val = []
        for cnt, (q,l) in enumerate(list(zip(questions_id,labels))):
            #print(q)
            #print(l)
            q,l = torch.from_numpy(np.array(q)), torch.from_numpy(np.array([l]))
            q,l = Variable(q).unsqueeze_(0), Variable(l)
            q.unsqueeze_(0)
            #print("Shapes of q and l: ", q.shape, l.shape)
            scores = net(q.squeeze_(0))
            scores_val.append(scores)
            #print("Shape of scores: ", scores.shape)
            c = scores.reshape(scores.shape[0],scores.shape[-1])            
            loss = criterion(c.float(),l.long())
            loss_val += loss.item()

            optimizer.zero_grad()
            loss.backward()       
            optimizer.step()

        scores_val = np.array(scores_val)

        print("Shape of scores_val: ", scores_val.shape, loss_val * 1.0 / 64)
        scores_val = np.random.random((64, 3001))
        #sys.exit()
 
        # compute accuracy
        predictions = np.argmax(scores_val, axis=1)
        accuracy = np.mean(np.logical_and(expr_validity_array,
                                          predictions == labels))
        avg_accuracy += (1-accuracy_decay) * (accuracy-avg_accuracy)
        validity = np.mean(expr_validity_array)

        print("Average Accuracy so far: ", avg_accuracy, n_iter)

run_training(max_iter, data_reader_trn)
