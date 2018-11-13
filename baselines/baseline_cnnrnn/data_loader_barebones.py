import nltk
import os
import torch
import sys
sys.path.append('/home/ubuntu/captions/')
import torch.utils.data as data
from vocabulary import Vocabulary
from PIL import Image
from pycocotools.coco import COCO
import numpy as np
from tqdm import tqdm
import random
import json


def collate_fn(batch):
    """Create batch"""

    #print("I got ", len(batch), " images in the batch")
    input_lengths = [len(x[1]) for x in batch]
    max_input_len = np.max(input_lengths) + 1

    a = [x[0] for x in batch ]
    b = np.array([ _pad(x[1], max_input_len)  for x in batch ], dtype=np.int)
    c = [x[2] for x in batch ]
    d = [x[3] for x in batch ]
    b_batch = torch.LongTensor(b)
    #print("I am returning images of length ", len(a))
    return a, b_batch, c, d

def _pad(seq, max_len):
    return np.pad(seq, (0, max_len - len(seq)),
                  mode='constant', constant_values=0)

def get_loader(transform,mode="train",batch_size=32,vocab_threshold=None,vocab_file="./vocab.pkl", start_word="<start>",end_word="<end>",
                                              unk_word="<unk>",vocab_from_file=True,num_workers=0,cocoapi_loc="/home/ubuntu/captions/"):

    assert mode in ["train", "val", "test"], "mode must be one of 'train', 'val' or 'test'."
    if vocab_from_file == False: 
        assert mode == "train", "To generate vocab from captions file, \
               must be in training mode (mode='train')."

    if mode == "train":
        if vocab_from_file == True: 
            assert os.path.exists(vocab_file), "vocab_file does not exist.  \
                   Change vocab_from_file to False to create vocab_file."
        assert batch_size==32

    img_folder = "/home/ubuntu/data/VQA/train2014"
    annotations_file = os.path.join(cocoapi_loc, "cocoapi/annotations/captions_train2014.json")
    if mode == "val":
        assert os.path.exists(vocab_file), "Must first generate vocab.pkl from training data."
        assert vocab_from_file == True, "Change vocab_from_file to True."
        img_folder = "/home/ubuntu/data/VQA/val2014"
        annotations_file = os.path.join(cocoapi_loc, "cocoapi/annotations/captions_val2014.json")  

    
    if mode == "test":
        assert batch_size == 1, "Please change batch_size to 1 if testing your model."
        assert os.path.exists(vocab_file), "Must first generate vocab.pkl from training data."
        assert vocab_from_file == True, "Change vocab_from_file to True."
        img_folder = "/home/ubuntu/data/VQA/test2014"
        annotations_file = os.path.join(cocoapi_loc, "cocoapi/annotations/image_info_test2014.json")

    
    dataset = CoCoDataset(transform=transform,mode=mode,batch_size=batch_size,vocab_threshold=vocab_threshold,vocab_file=vocab_file,start_word=start_word,
					end_word=end_word,unk_word=unk_word,annotations_file=annotations_file,vocab_from_file=vocab_from_file,img_folder=img_folder)
    

    if mode == "train":  
        indices = dataset.get_indices()
        initial_sampler = data.sampler.SubsetRandomSampler(indices=indices)
        data_loader = data.DataLoader(dataset=dataset, 
                                      num_workers=num_workers,
                                      batch_size=dataset.batch_size,
                                      collate_fn=collate_fn)
                                      #batch_sampler=data.sampler.BatchSampler(sampler=initial_sampler,
                                      #                                        batch_size=dataset.batch_size,
                                      #                                        drop_last=False))
    else:
        data_loader = data.DataLoader(dataset=dataset,
                                      batch_size=dataset.batch_size,
                                      shuffle=False,
                                      num_workers=num_workers,
                                      collate_fn=collate_fn)
    return data_loader




class CoCoDataset(data.Dataset):
    def __init__(self, transform, mode, batch_size, vocab_threshold, vocab_file, start_word, 
    end_word, unk_word, annotations_file, vocab_from_file, img_folder):
        self.transform = transform
        self.mode = mode
        self.batch_size = batch_size
        self.vocab = Vocabulary(vocab_threshold, vocab_file, start_word,
        end_word, unk_word, annotations_file, vocab_from_file)
        self.img_folder = img_folder
        if self.mode == "train" or self.mode == "val":
            self.coco = COCO(annotations_file)
            self.ids = list(self.coco.anns.keys())
            print("Obtaining caption lengths...")
            all_tokens = [nltk.tokenize.word_tokenize(
                          str(self.coco.anns[self.ids[index]]["caption"]).lower())
            for index in tqdm(np.arange(len(self.ids)))]
            self.caption_lengths = [len(token) for token in all_tokens]
        # If in test mode
        else:
            test_info = json.loads(open(annotations_file).read())
            self.paths = [item["file_name"] for item in test_info["images"]] 


    def __getitem__(self, index):
        if self.mode == "train" or self.mode == "val":
            ann_id = self.ids[index]
            caption = self.coco.anns[ann_id]["caption"]
            img_id = self.coco.anns[ann_id]["image_id"]
            path = self.coco.loadImgs(img_id)[0]["file_name"]
            image = Image.open(os.path.join(self.img_folder, path)).convert("RGB")
            image = self.transform(image)

            # Convert caption to tensor of word ids.
            tokens = nltk.tokenize.word_tokenize(str(caption).lower())
            caption_orig = caption
            caption = []
            caption.append(self.vocab(self.vocab.start_word))
            caption.extend([self.vocab(token) for token in tokens])
            caption.append(self.vocab(self.vocab.end_word))
            caption = torch.Tensor(caption).long()

            # Return pre-processed image and caption tensors
            return image, caption, path, caption_orig

        else:
            path = self.paths[index]
            print(os.path.join(self.img_folder, path))
            # Convert image to tensor and pre-process using transform
            PIL_image = Image.open(os.path.join(self.img_folder, path)).convert("RGB")
            orig_image = np.array(PIL_image)
            image = self.transform(PIL_image)

            # Return original image and pre-processed image tensor
            return orig_image, image, path




    def get_indices(self):
        sel_length = np.random.choice(self.caption_lengths)
        all_indices = np.where([self.caption_lengths[i] == \
                               sel_length for i in np.arange(len(self.caption_lengths))])[0]
        indices = list(np.random.choice(all_indices, size=self.batch_size))
        return indices

    def __len__(self):
        if self.mode == "train" or self.mode == "val":
            return len(self.ids)
        else:
            return len(self.paths)


