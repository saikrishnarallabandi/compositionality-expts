import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import model
import time
from torch.autograd import Variable
import torch.nn as nn


class wiki_dataset(Dataset):

   def __init__(self, train_path):
       self.train = np.load(train_path)

   def __getitem__(self, index):
       return self.train[index]

   def __len__(self):
       return len(self.train) 


train_file = '../dataset/wiki.train.npy'
valid_file = '../dataset/wiki.valid.npy'

wks_train = wiki_dataset(train_file)
train_loader = DataLoader(wks_train,
                          batch_size=1,
                          shuffle=True,
                          num_workers=4
                         )

def evaluate():
    wks_valid = wiki_dataset(valid_file)
    valid_loader = DataLoader(wks_valid,
                          batch_size=1,
                          shuffle=True,
                          num_workers=4)

    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    hidden = model.init_hidden(1)
    len_data = 0
    for i, t in enumerate(valid_loader):
        t = t.transpose(0, 1)
        len_data = len(t)
        data = t[0:len(t)-1].long().cuda()
        targets = t[1:].view(-1).long().cuda()
        del t

        data,targets = Variable(data,volatile=True), Variable(targets,volatile=True)
        output, hidden = model(data, None)
        loss = criterion(output.view(-1, 33278), targets)
        del data, targets, output

        total_loss += loss.data.cpu().numpy()    #len(data) * criterion(output_flat, targets).data
        hidden = repackage_hidden(hidden)
    return total_loss / float(i)



def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0
    start_time = time.time()
    hidden = model.init_hidden(1)
    for i, t in enumerate(train_loader):

        # Get the right shapes
        t = t.transpose(0, 1)
        data = t[0:len(t)-1].long().cuda()
        targets = t[1:].view(-1).long().cuda()
        del t

        # Get loss
        data,targets = Variable(data), Variable(targets)
        output, hidden = model(data, None)
        loss = criterion(output.view(-1, 33278), targets)
        del data, targets, output

        # Update model params
        model.zero_grad()
        loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm(model.parameters(),0.25)
        for p in model.parameters():
            p.data.add_(-lr, p.grad.data)
        total_loss += loss.data.cpu().numpy()
        del loss
        if i%100 == 1:
            print("Loss after ", i, "sequences: ", total_loss / float(i+0.00001))

#     def __init__(self, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):

criterion = nn.CrossEntropyLoss()
model = model.RNNLM(33278, 64, 128, 1, 0.3, 0)
model = model.cuda()
lr = 20

best_val_loss = None
for epoch in range(15):
   train()

   with open('model_epoch' + str(epoch).zfill(2) + '.pt', 'wb') as f:
           torch.save(model, f)

   val_loss = evaluate()
   print("Validation Loss: ", val_loss)
   if not best_val_loss or val_loss < best_val_loss:
        with open('model_best' + '.pt', 'wb') as f:
           torch.save(model, f)
        print("Saving the best model at epoch ", str(epoch))
        best_val_loss = val_loss
   else:
      print("Decreasing the learning rate")
      lr /= 4.0

