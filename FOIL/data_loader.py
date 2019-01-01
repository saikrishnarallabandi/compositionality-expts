import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os, sys
import pickle
import numpy as np
import nltk
from build_vocab import Vocabulary
import argparse

#{'image_id': 210286, 'target_word': '', 'foil_word': '', 'id': 376104, 'foil': True, 'foil_id': 2353800, 'caption': 'a man without a shirt, wearing a baseball frisbee tossing up a baseball.'}
#    item['image_name']=image_str
#    item['feature']=imageid2features[image_str]


class CocoDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(self, foil_data, vocab, transform=None):
        """Set the path for images, captions and vocabulary wrapper.
        
        Args:
            foil_data: foil_data
            vocab: vocabulary wrapper.
            transform: image transformer.
        """
        self.image_names = []
        self.features = []
        self.captions = []
        self.targets = []

        for i, item in enumerate(foil_data):
            self.image_names.append(item['image_name'])
            self.features.append(item['feature'])
            self.captions.append(item['caption'])
            self.targets.append(item['foil'])
        
            if (i % 1000) == 0:
                print("[{}/{}] curated.".format(i+1, len(self.image_names)))

        self.vocab = vocab

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        vocab = self.vocab
        caption = self.captions[index]
        img_feature = self.features[index]
        img_name = self.image_names[index]
        target = [0]
        if self.targets[index]:
            target = [1]

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        caption = torch.Tensor(caption)
        image = torch.Tensor(img_feature)
        target = torch.IntTensor(target)
        return image, caption, target, img_name

    def __len__(self):
        return len(self.image_names)


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
    images, captions, targets, image_names = zip(*data)

    images = torch.stack(images, 0)
    targets = torch.stack(targets, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    caps = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        caps[i, :end] = cap[:end]        
    
    return images, caps, lengths, targets, image_names


def get_loader(foil_dict, vocab, transform, batch_size, shuffle, num_workers):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # COCO caption dataset
    coco = CocoDataset(foil_data=foil_dict,
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

    with open(args.foil_pickle_file, 'rb') as f:
        foil_data = pickle.load(f)

    data_loader = get_loader(foil_dict=foil_data,
                             vocab=vocab, 
                             transform=None, 
                             batch_size=args.batch_size, 
                             shuffle=True, 
                             num_workers=2)

    for i, data_item in enumerate(data_loader): 
        img_features, cap_features, lens, targets, names = data_item
        print(img_features.size())
        print(cap_features.size())
        print(lens)
        print(targets.size())
        print(names)
        sys.exit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--foil_pickle_file', type=str,
                        default='./foil_train_with_features.pkl',
                        help='path for foil training data file')

    parser.add_argument('--vocab_path', type=str, default='./vocab.pkl',
                        help='path for vocabulary file')

    parser.add_argument('--batch_size', type=int, default=16,
                        help='batch size')

    args = parser.parse_args()
    main(args)
