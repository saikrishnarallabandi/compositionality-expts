import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os, sys
import pickle
import numpy as np
import nltk
from build_vocab import Vocabulary
import argparse


class CocoDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(self, i2f_dict, i2c_dict, vocab, transform=None):
        """Set the path for images, captions and vocabulary wrapper.
        
        Args:
            i2f_dict: image id to feature map
            i2c_dict: image id to caption map
            vocab: vocabulary wrapper.
            transform: image transformer.
        """
        self.ids = i2f_dict.keys()
        self.image_name = []
        self.features = []
        self.captions = []

        self.image_ids = list(self.ids)
        for i, id in enumerate(self.ids):
          if i > len(self.ids) - 3:
            continue
          else:
            id_next = self.image_ids[i+1]
            self.image_name.append(id)
            self.features.append(i2f_dict[id])
            self.captions.append(i2c_dict[id_next])
        
            if (i % 1000) == 0:
                print("[{}/{}] curated.".format(i+1, len(self.ids)))

        self.vocab = vocab

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        vocab = self.vocab
        caption = self.captions[index]
        img_feature = self.features[index]
        img_name = self.image_name[index]

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        image = torch.Tensor(img_feature)
        return image, target, img_name

    def __len__(self):
        return len(self.ids)


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption, image_name).
    
    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.
    Args:
        data: list of tuple (image, caption, image_name). 
            - image: torch tensor of shape (features_dim).
            - caption: torch tensor of shape (?); variable length.
    Returns:
        images: torch tensor of shape (batch_size, features_dim).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions, image_names = zip(*data)

    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]        
    
    return images, targets, lengths, image_names


def get_loader(i2f_dict, i2c_dict, vocab, transform, batch_size, shuffle, num_workers):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # COCO caption dataset
    coco = CocoDataset(i2f_dict=i2f_dict,
                       i2c_dict=i2c_dict,
                       vocab=vocab,
                       transform=transform)
    
    # Data loader for COCO dataset
    # This will return (images, captions, lengths, image_names) for each iteration.
    # images: a tensor of shape (batch_size, feature_lenght).
    # captions: a tensor of shape (batch_size, padded_length).
    # lengths: a list indicating valid length for each caption. length is (batch_size).
    data_loader = torch.utils.data.DataLoader(dataset=coco, 
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader


def main(args):
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    with open(args.imgid2caption_pickle_file, 'rb') as f:
        imageid2captions = pickle.load(f)

    with open(args.imgid2feature_pickle_file, 'rb') as f:
        imageid2features = pickle.load(f)

    data_loader = get_loader(i2f_dict=imageid2features,
                             i2c_dict=imageid2captions, 
                             vocab=vocab, 
                             transform=None, 
                             batch_size=args.batch_size, 
                             shuffle=True, 
                             num_workers=2)

    for i, data_item in enumerate(data_loader): 
        img_features, cap_features, lens, names = data_item
        print(img_features.size())
        print(cap_features.size())
        print(lens)
        print(names)
        sys.exit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgid2caption_pickle_file', type=str,
                        default='./imageid2captions.pkl',
                        help='path for image id 2 captions file')

    parser.add_argument('--imgid2feature_pickle_file', type=str,
                        default='./imageid2features.pkl',
                        help='path for image id 2 features file')

    parser.add_argument('--vocab_path', type=str, default='./vocab.pkl',
                        help='path for vocabulary file')

    parser.add_argument('--batch_size', type=int, default=16,
                        help='batch size')

    args = parser.parse_args()
    main(args)
