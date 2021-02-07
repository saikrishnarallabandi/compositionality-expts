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

sys.path.append('/home/ubuntu/captions/')

#from utils_barebones import train, validate, save_epoch, early_stopping
from data_loader_barebones import get_loader
from model_barebones import EncoderCNN, DecoderRNN

start_time = time.time()

# Set values for the training variables
batch_size = 32         # batch size
vocab_threshold = 5     # minimum word count threshold
vocab_from_file = True  # if True, load existing vocab file
embed_size = 256        # dimensionality of image and word embeddings
hidden_size = 512       # number of features in hidden state of the RNN decoder
num_epochs = 10          # number of training epochs

with open('train_loader.pkl', 'rb') as handle:
    train_loader = pickle.load(handle)

with open('valid_loader.pkl', 'rb') as handle:
    val_loader = pickle.load(handle)

print("Loading using pkl files took ", time.time() - start_time)

# The size of the vocabulary
vocab_size = len(train_loader.dataset.vocab)

# Initialize the encoder and decoder
encoder = EncoderCNN(embed_size)
decoder = DecoderRNN(embed_size, hidden_size, vocab_size)

# Move models to GPU if CUDA is available
if torch.cuda.is_available():
    encoder.cuda()
    decoder.cuda()



# Define the loss function
criterion = nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()

# Specify the learnable parameters of the model
params = list(decoder.parameters()) + list(encoder.embed.parameters()) + list(encoder.bn.parameters())

# Define the optimizer
optimizer = torch.optim.Adam(params=params, lr=0.001)



# Set the total number of training and validation steps per epoch
total_train_step = math.ceil(len(train_loader.dataset.caption_lengths) / train_loader.batch_sampler.batch_size)
total_val_step = math.ceil(len(val_loader.dataset.caption_lengths) / val_loader.batch_sampler.batch_size)
print ("Number of training steps:", total_train_step)
print ("Number of validation steps:", total_val_step)


# Keep track of train and validation losses and validation Bleu-4 scores by epoch
train_losses = []
val_losses = []
val_bleus = []
# Keep track of the current best validation Bleu score
best_val_bleu = float("-INF")

start_time = time.time()
for epoch in range(1, num_epochs + 1):
    train_loss = train(train_loader, encoder, decoder, criterion, optimizer, 
                       vocab_size, epoch, total_train_step)
    train_losses.append(train_loss)
    val_loss, val_bleu = validate(val_loader, encoder, decoder, criterion,
                                  train_loader.dataset.vocab, epoch, total_val_step)
    val_losses.append(val_loss)
    val_bleus.append(val_bleu)
    if val_bleu > best_val_bleu:
        print ("Validation Bleu-4 improved from {:0.4f} to {:0.4f}, saving model to best-model.pkl".
               format(best_val_bleu, val_bleu))
        best_val_bleu = val_bleu
        filename = os.path.join("./models", "best-model.pkl")
        save_epoch(filename, encoder, decoder, optimizer, train_losses, val_losses, 
                   val_bleu, val_bleus, epoch)
    else:
        print ("Validation Bleu-4 did not improve, saving model to model-{}.pkl".format(epoch))
    # Save the entire model anyway, regardless of being the best model so far or not
    filename = os.path.join("./models-train", "model-{}.pkl".format(epoch))
    save_epoch(filename, encoder, decoder, optimizer, train_losses, val_losses, 
               val_bleu, val_bleus, epoch)
    print ("Epoch [%d/%d] took %ds" % (epoch, num_epochs, time.time() - start_time))
    if epoch > 5:
        # Stop if the validation Bleu doesn't improve for 3 epochs
        if early_stopping(val_bleus, 3):
            break
    start_time = time.time()


# Load the last checkpoints
checkpoint = torch.load(os.path.join('./models-train', 'train-model-76500.pkl'))

# Load the pre-trained weights
encoder.load_state_dict(checkpoint['encoder'])
decoder.load_state_dict(checkpoint['decoder'])
optimizer.load_state_dict(checkpoint['optimizer'])

# Load start_loss from checkpoint if in the middle of training process; otherwise, comment it out
start_loss = checkpoint['total_loss']
# Reset start_loss to 0.0 if starting a new epoch; otherwise comment it out
#start_loss = 0.0

# Load epoch. Add 1 if we start a new epoch
epoch = checkpoint['epoch']
# Load start_step from checkpoint if in the middle of training process; otherwise, comment it out
start_step = checkpoint['train_step'] + 1
# Reset start_step to 1 if starting a new epoch; otherwise comment it out
#start_step = 1

# Train 1 epoch at a time due to very long training time
train_loss = train(val_loader, encoder, decoder, criterion, optimizer, 
                   vocab_size, epoch, total_train_step, start_step, start_loss)



print("Train Loss after 100 batches is ", train_loss)


def train(train_loader, ved_model, criterion, optimizer, vocab_size,
          epoch, total_step, start_step=1, start_loss=0.0):
    """Train the model for one epoch using the provided parameters. Save 
    checkpoints every 100 steps. Return the epoch's average train loss."""

    # Switch to train mode
    ved_model.train()
    #logger = Logger('./logs')
    # Keep track of train loss
    total_loss = start_loss

    # Start time for every 100 steps
    start_train_time = time.time()

    for i_step in range(start_step, total_step + 1):
        # Randomly sample a caption length, and sample indices with that length
        indices = train_loader.dataset.get_indices()
        # Create a batch sampler to retrieve a batch with the sampled indices
        new_sampler = data.sampler.SubsetRandomSampler(indices=indices)
        train_loader.batch_sampler.sampler = new_sampler

        # Obtain the batch
        for batch in train_loader:
            images, captions = batch[0], batch[1]
            break 
        
        # Move to GPU if CUDA is available
        if torch.cuda.is_available():
            images = images.cuda()
            captions = captions.cuda()
        print("Shape of images: ", images.shape)
        # Pass the inputs through the VED model
        outputs,mu,logvar = ved_model(images, captions)

        # Calculate the batch loss
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        ce = criterion(outputs.view(-1, vocab_size), captions.view(-1)) 
        loss = ce + KLD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if i_step % 50 == 0:
            print("Epoch %d, Train step [%d/%d], %ds, CE Loss: %.4f, KL Loss: %4f, Perplexity: %5.4f" \
                % (epoch, i_step, total_step, time.time() - start_train_time,
                   ce.item(), KLD.item(), np.exp(loss.item())))

    return total_loss / total_step
            
def validate(val_loader, ved_model, criterion, vocab, epoch, 
             total_step, start_step=1, start_loss=0.0, start_bleu=0.0):
    """Validate the model for one epoch using the provided parameters. 
    Return the epoch's average validation loss and Bleu-4 score."""

    # Switch to validation mode
    ved_model.eval()

    # Initialize smoothing function
    smoothing = SmoothingFunction()

    # Keep track of validation loss and Bleu-4 score
    total_loss = start_loss
    total_bleu_4 = start_bleu

    # Start time for every 100 steps
    start_val_time = time.time()

    # Disable gradient calculation because we are in inference mode
    with torch.no_grad():
        for i_step in range(start_step, total_step + 1):
            # Randomly sample a caption length, and sample indices with that length
            indices = val_loader.dataset.get_indices()
            # Create a batch sampler to retrieve a batch with the sampled indices
            new_sampler = data.sampler.SubsetRandomSampler(indices=indices)
            val_loader.batch_sampler.sampler = new_sampler

            # Obtain the batch
            for batch in val_loader:
                images, captions = batch[0], batch[1]
                break 

            # Move to GPU if CUDA is available
            if torch.cuda.is_available():
                images = images.cuda()
                captions = captions.cuda()
            
            # Pass the inputs through the CNN-RNN model
            # features from encoder are batch X feat_size [32 X 256]
            #features = encoder(images)
            #print(features)
            #print(features.shape, "features size before entering decoder")
            #print(exit(1))
            #outputs = decoder(features, captions)
            outputs, mu, logvar = ved_model(images, captions)

            # Calculate the total Bleu-4 score for the batch
            batch_bleu_4 = 0.0
            # Iterate over outputs. Note: outputs[i] is a caption in the batch
            # outputs[i, j, k] contains the model's predicted score i.e. how 
            # likely the j-th token in the i-th caption in the batch is the 
            # k-th token in the vocabulary.
            for i in range(len(outputs)):
                predicted_ids = []
                for scores in outputs[i]:
                    # Find the index of the token that has the max score
                    predicted_ids.append(scores.argmax().item())
                # Convert word ids to actual words
                predicted_word_list = word_list(predicted_ids, vocab)
                caption_word_list = word_list(captions[i].cpu().numpy(), vocab)
                # Calculate Bleu-4 score and append it to the batch_bleu_4 list
                batch_bleu_4 += sentence_bleu([caption_word_list], 
                                               predicted_word_list, 
                                               smoothing_function=smoothing.method1)
            total_bleu_4 += batch_bleu_4 / len(outputs)

            # Calculate the batch loss
            KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            ce = criterion(outputs.view(-1, len(vocab)), captions.view(-1))
            loss = ce + KLD
            total_loss += loss.item()

            if i_step % 50 == 0:
                print("Valid step [%d/%d], %ds, CE Loss: %.4f, KL Loss: %4f, Perplexity: %5.4f" \
                     % (i_step, total_step, time.time() - start_val_time,
                        ce.item(), KLD.item(), np.exp(loss.item())))
        
            # Get validation statistics
            #stats = "Epoch %d, Val step [%d/%d], %ds, Loss: %.4f, Perplexity: %5.4f, Bleu-4: %.4f" \
            #        % (epoch, i_step, total_step, time.time() - start_val_time,
            #           loss.item(), np.exp(loss.item()), batch_bleu_4 / len(outputs))

            # Print validation statistics (on same line)
            #print("\r" + stats, end="")
            #sys.stdout.flush()

            # Print validation statistics (on different line) and reset time
            #if i_step % PRINT_EVERY == 0:
            #    print("\r" + stats)
            #    filename = os.path.join("./models", "val-model-{}{}.pkl".format(epoch, i_step))
            #    #save_val_checkpoint(filename, encoder, decoder, total_loss, total_bleu_4, epoch, i_step)
            #    start_val_time = time.time()
                
        return total_loss / total_step, total_bleu_4 / total_step

def save_checkpoint(filename, encoder, decoder, optimizer, total_loss, epoch, train_step=1):
    """Save the following to filename at checkpoints: encoder, decoder,
    optimizer, total_loss, epoch, and train_step."""
    torch.save({"encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
                "optimizer" : optimizer.state_dict(),
                "total_loss": total_loss,
                "epoch": epoch,
                "train_step": train_step,
               }, filename)

def save_val_checkpoint(filename, encoder, decoder, total_loss,
    total_bleu_4, epoch, val_step=1):
    """Save the following to filename at checkpoints: encoder, decoder,
    total_loss, total_bleu_4, epoch, and val_step"""
    torch.save({"encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
                "total_loss": total_loss,
                "total_bleu_4": total_bleu_4,
                "epoch": epoch,
                "val_step": val_step,
               }, filename)

def save_epoch(filename, encoder, decoder, optimizer, train_losses, val_losses, 
               val_bleu, val_bleus, epoch):
    """Save at the end of an epoch. Save the model's weights along with the 
    entire history of train and validation losses and validation bleus up to 
    now, and the best Bleu-4."""
    torch.save({"encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
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
    sentence = []
    for i in range(len(word_idx_list)):
        vocab_id = word_idx_list[i]
        word = vocab.idx2word[vocab_id]
        if word == vocab.end_word:
            break
        if word != vocab.start_word:
            sentence.append(word)
    sentence = " ".join(sentence)
    return sentence

def get_prediction(data_loader, encoder, decoder, vocab):
    """Loop over images in a dataset and print model's top three predicted 
    captions using beam search."""
    orig_image, image = next(iter(data_loader))
   # plt.imshow(np.squeeze(orig_image))
   # plt.title("Sample Image")
   # plt.show()
    if torch.cuda.is_available():
        image = image.cuda()
    features = encoder(image).unsqueeze(1)
    print ("Caption without beam search:")
    output = decoder.sample(features)
    sentence = clean_sentence(output, vocab)
    print (sentence)

    print ("Top captions using beam search:")
    outputs = decoder.sample_beam_search(features)
    # Print maximum the top 3 predictions
    num_sents = min(len(outputs), 3)
    for output in outputs[:num_sents]:
        sentence = clean_sentence(output, vocab)
        print (sentence)

