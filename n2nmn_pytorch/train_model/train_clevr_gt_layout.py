from __future__ import absolute_import, division, print_function
from Utils.data_reader import DataReader
import sys
from torch import optim

from models.layout_assembler import Assembler
from train_model.input_parameters import *
from models.end2endModuleNet import *
from models.custom_loss import custom_loss
from global_variables.global_variables import use_cuda

from logger import Logger


##create directory for snapshot
os.makedirs(snapshot_dir, exist_ok=True)

##create tensorboard logger
logger_object = Logger('./logging')

assembler = Assembler(vocab_layout_file)

data_reader_trn = DataReader(imdb_file_trn,image_feat_dir, shuffle=True, one_pass=False,
                             batch_size=N,
                             T_encoder=T_encoder,
                             T_decoder=T_decoder,
                             assembler=assembler,
                             vocab_question_file=vocab_question_file,
                             vocab_answer_file=vocab_answer_file,
                             prune_filter_module=prune_filter_module,
                             C_encoder = C_encoder, 
                             C_decoder = C_decoder,
                             vocab_captions_file = vocab_captions_file)

num_vocab_txt = data_reader_trn.batch_loader.vocab_dict.num_vocab
num_vocab_nmn = len(assembler.module_names)
num_choices = data_reader_trn.batch_loader.answer_dict.num_vocab
num_c_vocab_txt = data_reader_trn.batch_loader.caption_dict.num_vocab

criterion_layout = custom_loss(lambda_entropy = lambda_entropy)
criterion_answer = nn.CrossEntropyLoss(size_average=False,reduce=False)


if model_type == model_type_gt_rl:
    myModel = torch.load(model_path)
else:
    myModel = end2endModuleNet(num_vocab_txt=num_vocab_txt, num_vocab_nmn=num_vocab_nmn, out_num_choices=num_choices,
                               embed_dim_nmn=embed_dim_nmn, embed_dim_txt=embed_dim_txt,
                               image_height=H_feat, image_width=W_feat, in_image_dim=D_feat,
                               hidden_size=lstm_dim, assembler=assembler, layout_criterion=criterion_layout,
                               max_layout_len=T_decoder,
                               answer_criterion=criterion_answer, num_layers=num_layers, decoder_dropout=0, 
                               num_c_vocab_txt=num_c_vocab_txt)

myOptimizer = optim.Adam(myModel.parameters(), weight_decay=weight_decay, lr=learning_rate)


avg_accuracy = 0
accuracy_decay = 0.99
avg_layout_accuracy = 0
updated_baseline = np.log(28)
count = 0

#def start_traing_model(epoch_number):
for epoch_number in range(10):
  for i_iter, batch in enumerate(data_reader_trn.batches()):
    if i_iter >= max_iter:
        break

    _, n_sample = batch['input_seq_batch'].shape
    input_text_seq_lens = batch['seq_length_batch']
    input_text_seqs = batch['input_seq_batch']
    input_layouts = batch['gt_layout_batch']
    input_images = batch['image_feat_batch']
    input_answers = batch['answer_label_batch']
    input_caption_seqs = batch['input_c_seq_batch']
    input_caption_seq_lens = batch['c_seq_length_batch']
    
    n_correct_layout = 0
    n_correct_answer = 0

    input_txt_variable = Variable(torch.LongTensor(input_text_seqs))
    input_txt_variable = input_txt_variable.cuda() if use_cuda else input_txt_variable

    input_cap_variable = Variable(torch.LongTensor(input_caption_seqs))
    input_cap_variable = input_cap_variable.cuda() if use_cuda else input_cap_variable

    input_layout_variable = None

    if model_type == model_type_gt:
        input_layout_variable = Variable(torch.LongTensor(input_layouts))
        input_layout_variable = input_layout_variable.cuda() if use_cuda else input_layout_variable
    
    myOptimizer.zero_grad()

    total_loss,avg_answer_loss ,myAnswer, predicted_layouts, expr_validity_array, updated_baseline \
        = myModel(input_txt_variable=input_txt_variable, input_text_seq_lens=input_text_seq_lens,
                  input_answers=input_answers, input_images=input_images,policy_gradient_baseline=updated_baseline,
                  baseline_decay=baseline_decay, input_layout_variable=input_layout_variable,
                  sample_token=decoder_sampling, input_cap_variable=input_cap_variable, 
                  input_cap_seq_lens=input_caption_seq_lens)

    if total_loss is not None:
        total_loss.backward()
        torch.nn.utils.clip_grad_norm(myModel.parameters(), max_grad_l2_norm)
        myOptimizer.step()

    layout_accuracy = np.mean(np.all(predicted_layouts == input_layouts, axis=0))
    avg_layout_accuracy += (1 - accuracy_decay) * (layout_accuracy - avg_layout_accuracy)

    accuracy = np.mean(np.logical_and(expr_validity_array, myAnswer == input_answers))
    avg_accuracy += (1 - accuracy_decay) * (accuracy - avg_accuracy)
    validity = np.mean(expr_validity_array)

    if (i_iter + 1) % 20 == 0 :
        logger_object.scalar_summary("VQA Accuracy", avg_accuracy, count)
        logger_object.scalar_summary("Layout Accuracy", avg_layout_accuracy, count)
        logger_object.scalar_summary("Loss",  avg_answer_loss.data.cpu().numpy(), count)
        print("Epoch: ", epoch_number, " iter:", i_iter + 1,
              " cur_layout_acc:%.3f"% layout_accuracy, " avg_layout_acc:%.3f"% avg_layout_accuracy,
              " cur_ans_acc:%.4f"% accuracy, " avg_answer_acc:%.4f"% avg_accuracy,
              "total loss:%.4f"%total_loss.data.cpu().numpy(),
              "avg_answer_loss:%.4f"% avg_answer_loss.data.cpu().numpy())

        sys.stdout.flush()

    # Save snapshot
    if  (i_iter + 1) % snapshot_interval == 0 or (i_iter + 1) == max_iter:
        model_snapshot_file = os.path.join(snapshot_dir, "model_%08d" % (i_iter + 1))
        torch.save(myModel, model_snapshot_file)
        print('snapshot saved to ' + model_snapshot_file )
        sys.stdout.flush()
    count = count + 1


#start_traing_model(0)
