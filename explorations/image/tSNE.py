import os
import random
import numpy as np
import json
import matplotlib.pyplot as plt

import pickle as pickle
from matplotlib.pyplot import imshow
from PIL import Image
from sklearn.manifold import TSNE
from tqdm import tqdm
from IPython.display import clear_output
import glob
from os.path import basename
import torch.nn.functional as F
import torch

all_files = glob.glob('./images/train/*.png') #('./train2014/*.png')
images = []
features = []

rand_list = random.sample(range(0, len(all_files)), 5000)

count = 0
for i in rand_list:
    if (count+1)%100 == 0:
        print("Completed ", count, "/15000")
    file_name = basename(all_files[i]).split('.')[0]
    
    images.append(all_files[i])
    feature_file = './vgg_pool5_us/train/'+file_name+'.npy'
   
    feature_St = np.load(feature_file)
    feature = feature_St.transpose(0,3,1,2)
    feature = torch.from_numpy(feature)
    feature_output = F.avg_pool2d(feature, feature.shape[2], stride=1)
    feature_output = feature_output.numpy()
    feature = feature_output.reshape(-1,1)
    features.append(feature)
    count = count + 1

print("\n\n\n Starting tSNE \n\n\n")
X = np.array(features).squeeze()
print(X.shape)
tsne = TSNE(n_components=2, learning_rate=150, perplexity=30, angle=0.2, verbose=2).fit_transform(X)

tx, ty = tsne[:,0], tsne[:,1]
tx = (tx-np.min(tx)) / (np.max(tx) - np.min(tx))
ty = (ty-np.min(ty)) / (np.max(ty) - np.min(ty))

width = 4000
height = 3000
max_dim = 100

full_image = Image.new('RGBA', (width, height))
for img, x, y in tqdm(zip(images, tx, ty)):
    tile = Image.open(img)
    rs = max(1, tile.width/max_dim, tile.height/max_dim)
    tile = tile.resize((int(tile.width/rs), int(tile.height/rs)), Image.ANTIALIAS)
    full_image.paste(tile, (int((width-max_dim)*x), int((height-max_dim)*y)), mask=tile.convert('RGBA'))

plt.figure(figsize = (16,12))
imshow(full_image)

full_image.save("./tSNE-train_CLEVR_vgg_pool5_us.png")
