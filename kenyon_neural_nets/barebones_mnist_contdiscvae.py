import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np

bsz = 128
max_epochs = 100
beta = 4
EPS = 1e-4

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=bsz, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=bsz, shuffle=True)


class VAE(nn.Module):

   def __init__(self):
       super(VAE, self).__init__()

       self.encoder_fc = nn.Linear(784, 400)
       self.mean_fc = nn.Linear(400, 2000)
       self.logvar_fc = nn.Linear(400,2000)
       self.discretez_fc = nn.Linear(400, 1000)
       self.prefinal_fc = nn.Linear(30, 400)
       self.final_fc = nn.Linear(400, 784)

   def encoder(self, x):
       encoded = torch.relu(self.encoder_fc(x))
       return encoded

   def reparameterize_continuous(self, encoded):
        mu = self.mean_fc(encoded)
        log_var = self.logvar_fc(encoded)

        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)

        return eps.mul(std).add_(mu), mu, log_var

   def reparameterize_discrete(self, encoded):
        alpha = self.discretez_fc(encoded)
        #print("Shape of alpha: ", alpha.shape)
        alpha = F.softmax(alpha, dim=0)
        #print("Shape of alpha: ", alpha.shape)
        return alpha

   def decoder(self, z):
        decoded = F.relu(self.prefinal_fc(z))
        return torch.sigmoid(self.final_fc(decoded))

   def forward(self, x):
        x = x.view(-1, 784)
        encoded = self.encoder(x)
        z_continuous, mu, log_var = self.reparameterize_continuous(encoded)
        z_discrete = self.reparameterize_discrete(encoded)
        z = torch.cat([z_continuous, z_discrete], dim=1)
        #print("Shape of z from reparameterization: ", z_continuous.shape, z_discrete.shape, z.shape)
        return self.decoder(z), mu, log_var, z_discrete


model = VAE()
if torch.cuda.is_available():
  model = model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)



# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function_continuous(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + beta * KLD

def loss_function_discrete(alpha):
    disc_dim = int(alpha.size()[-1])
    log_dim = torch.Tensor([np.log(disc_dim)])
    if torch.cuda.is_available():
       log_dim = log_dim.cuda()
    neg_entropy = torch.sum(alpha * torch.log(alpha + EPS), dim=1)
    mean_neg_entropy = torch.mean(neg_entropy, dim=0)
    kl_loss = log_dim + mean_neg_entropy
    return kl_loss

def val(epoch):

   model.eval()
   z = torch.rand(1,30).cuda()
   with torch.no_grad():
      a = model.decoder(z).view(1,1,28,28)
      save_image(a.cpu(),'results_betavae_contdisc/reconstruction_' + str(epoch) + '.png')
      

def train(epoch):

   model.train()
   train_loss = 0
 
   for i, (data, _) in enumerate(train_loader):

       if torch.cuda.is_available():
          data = data.cuda()
       optimizer.zero_grad()
       recon_batch, mu, logvar, alpha = model(data)
       loss_continuous = loss_function_continuous(recon_batch, data, mu, logvar)
       loss_discrete = loss_function_discrete(alpha)
       loss = loss_continuous + loss_discrete
       loss.backward()
       train_loss += loss.item()
       torch.nn.utils.clip_grad_norm(model.parameters(), 0.25)
       optimizer.step()

       #if i % 10 == 1:
       #   print(train_loss/(i+1))

   return train_loss/(len(train_loader.dataset))



def main():

  for epoch in range(max_epochs):
      train_loss =  train(epoch)
      val(epoch)
      print("Train loss after ", epoch, " epochs: ", train_loss)


main()
