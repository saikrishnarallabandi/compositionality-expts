import os
from pycocotools.coco import COCO
from torchvision import transforms
import torch
import numpy as np

from data_loader import get_loader
from model import EncoderCNN, DecoderRNN
from utils import clean_sentence, get_prediction


transform_test = transforms.Compose([ 
    transforms.Resize(256),                          # smaller edge of image resized to 256
    transforms.CenterCrop(224),                      # get 224x224 crop from the center
    transforms.ToTensor(),                           # convert the PIL Image to a tensor
    transforms.Normalize((0.485, 0.456, 0.406),      # normalize image for pre-trained model
                         (0.229, 0.224, 0.225))])

# Create the data loader
data_loader = get_loader(transform=transform_test,    
                         mode='test')

orig_image, image = next(iter(data_loader))
# Convert image from torch.FloatTensor to numpy ndarray
transformed_image = image.numpy()
# Remove the first dimension which is batch_size euqal to 1
transformed_image = np.squeeze(transformed_image)
transformed_image = transformed_image.transpose((1, 2, 0))


checkpoint = torch.load(os.path.join('./models-val', 'model-1.pkl'))

# Specify values for embed_size and hidden_size - we use the same values as in training step
embed_size = 256
hidden_size = 512

# Get the vocabulary and its size
vocab = data_loader.dataset.vocab
vocab_size = len(vocab)

# Initialize the encoder and decoder, and set each to inference mode
encoder = EncoderCNN(embed_size)
encoder.eval()
decoder = DecoderRNN(embed_size, hidden_size, vocab_size)
decoder.eval()

# Load the pre-trained weights
encoder.load_state_dict(checkpoint['encoder'])
decoder.load_state_dict(checkpoint['decoder'])

# Move models to GPU if CUDA is available.
if torch.cuda.is_available():
    encoder.cuda()
    decoder.cuda()



get_prediction(data_loader, encoder, decoder, vocab)
get_prediction(data_loader, encoder, decoder, vocab)
get_prediction(data_loader, encoder, decoder, vocab)
get_prediction(data_loader, encoder, decoder, vocab)
get_prediction(data_loader, encoder, decoder, vocab)
get_prediction(data_loader, encoder, decoder, vocab)
get_prediction(data_loader, encoder, decoder, vocab)
get_prediction(data_loader, encoder, decoder, vocab)
get_prediction(data_loader, encoder, decoder, vocab)
get_prediction(data_loader, encoder, decoder, vocab)

