import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import sys
from pycocotools.coco import COCO
import math
import torch.utils.data as data
import numpy as np
import os
import requests
import time
import pickle


def save_checkpoint(filename, ved, optimizer, total_loss, epoch, train_step=1):
    """Save the following to filename at checkpoints: encoder, decoder,
    optimizer, total_loss, epoch, and train_step."""
    torch.save({"model": ved.state_dict(),
                "optimizer" : optimizer.state_dict(),
                "total_loss": total_loss,
                "epoch": epoch,
                "train_step": train_step,
               }, filename)


def save_val_checkpoint(filename, ved, total_loss,
    total_bleu_4, epoch, val_step=1):
    """Save the following to filename at checkpoints: encoder, decoder,
    total_loss, total_bleu_4, epoch, and val_step"""
    torch.save({"model": ved.state_dict(),
                "total_loss": total_loss,
                "total_bleu_4": total_bleu_4,
                "epoch": epoch,
                "val_step": val_step,
               }, filename)


def save_epoch(filename, ved, optimizer, train_losses, val_losses,
               val_bleu, val_bleus, epoch):
    """Save at the end of an epoch. Save the model's weights along with the
    entire history of train and validation losses and validation bleus up to
    now, and the best Bleu-4."""
    torch.save({"model": ved.state_dict(),
                "optimizer": optimizer.state_dict(),
                "train_losses": train_losses,
                "val_losses": val_losses,
                "val_bleu": val_bleu,
                "val_bleus": val_bleus,
                "epoch": epoch
               }, filename)


def early_stopping(val_bleus, patience=3):
    """Check if the validation Bleu-4 scores no longer improve for 3
    (or a specified number of) consecutive epochs."""
    # The number of epochs should be at least patience before checking
    # for convergence
    if patience > len(val_bleus):
        return False
    latest_bleus = val_bleus[-patience:]
    # If all the latest Bleu scores are the same, return True
    if len(set(latest_bleus)) == 1:
        return True
    max_bleu = max(val_bleus)
    if max_bleu in latest_bleus:
        # If one of recent Bleu scores improves, not yet converged
        if max_bleu not in val_bleus[:len(val_bleus) - patience]:
            return False
        else:
            return True
    # If none of recent Bleu scores is greater than max_bleu, it has converged
    return True


def word_list(word_idx_list, vocab):
    """Take a list of word ids and a vocabulary from a dataset as inputs
    and return the corresponding words as a list.
    """
    word_list = []
    for i in range(len(word_idx_list)):
        vocab_id = word_idx_list[i]
        word = vocab.idx2word[vocab_id]
        if word == vocab.end_word:
            break
        if word != vocab.start_word:
            word_list.append(word)
    return word_list


def clean_sentence(word_idx_list, vocab):
    """Take a list of word ids and a vocabulary from a dataset as inputs
    and return the corresponding sentence (as a single Python string).
    """

    word_idx_list = np.asarray(word_idx_list)
    word_idx_list = np.squeeze(word_idx_list)
    if len(word_idx_list.shape) == 1:
        word_idx_list = np.expand_dims(word_idx_list, 1)

    a, batch = word_idx_list.shape

    sentences = []
    for b in range(batch):
        sentence = []
        for i in range(len(word_idx_list)):
            vocab_id = word_idx_list[i][b]
            word = vocab.idx2word[vocab_id]
            if word == vocab.end_word:
                break
            if word != vocab.start_word:
                sentence.append(word)
        sentence = " ".join(sentence)
        sentences.append(sentence)
    return sentences
