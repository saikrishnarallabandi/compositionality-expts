
import numpy as np
import torch
from data_loader import get_loader
from torchvision import transforms
from collections import Counter
import torch.utils.data as data
from model import EncoderCNN, DecoderRNN

transform_train = transforms.Compose([ 
    transforms.Resize(256),                          # smaller edge of image resized to 256
    transforms.RandomCrop(224),                      # get 224x224 crop from random location
    transforms.RandomHorizontalFlip(),               # horizontally flip image with probability=0.5
    transforms.ToTensor(),                           # convert the PIL Image to a tensor
    transforms.Normalize((0.485, 0.456, 0.406),      # normalize image for pre-trained model
                         (0.229, 0.224, 0.225))])

# Set the minimum word count threshold.
vocab_threshold = 5

# Specify the batch size.
batch_size = 10

# Obtain the data loader.
data_loader = get_loader(transform=transform_train,
                         mode='train',
                         batch_size=batch_size,
                         vocab_threshold=vocab_threshold,
                         vocab_from_file=True)


counter = Counter(data_loader.dataset.caption_lengths)
lengths = sorted(counter.items(), key=lambda pair: pair[1], reverse=True)
for value, count in lengths:
    print('value: %2d --- count: %5d' % (value, count))


indices = data_loader.dataset.get_indices()
print('{} sampled indices: {}'.format(len(indices), indices))
# Create and assign a batch sampler to retrieve a batch with the sampled indices.
new_sampler = data.sampler.SubsetRandomSampler(indices=indices)
data_loader.batch_sampler.sampler = new_sampler
for batch in data_loader:
    images, captions = batch[0], batch[1]
    break


print('images.shape:', images.shape)
print('captions.shape:', captions.shape)


embed_size = 256
encoder = EncoderCNN(embed_size)
if torch.cuda.is_available():
    encoder = encoder.cuda()
if torch.cuda.is_available():
    images = images.cuda()

features = encoder(images)
print('type(features):', type(features))
print('features.shape:', features.shape)

assert (features.shape[0]==batch_size) & (features.shape[1]==embed_size), "The shape of the encoder output is incorrect."

