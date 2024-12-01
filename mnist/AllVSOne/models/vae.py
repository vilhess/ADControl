import torch 
import torch.nn as nn 

class Encoder(nn.Module):
    def __init__(self, in_dim, hidden_dims, latent_dim):
        super(Encoder, self).__init__()
        
        self.input_hidden = nn.Linear(in_dim, hidden_dims[0])
        self.hidden_hidden = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.hidden_mu = nn.Linear(hidden_dims[1], latent_dim)
        self.hidden_sigma = nn.Linear(hidden_dims[1], latent_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        h = self.input_hidden(x)
        h = self.relu(h)
        h = self.hidden_hidden(h)
        h = self.relu(h)

        mu = self.hidden_mu(h)
        logvar = self.hidden_sigma(h)

        return mu, logvar
    
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dims, out_dim):
        super(Decoder, self).__init__()

        self.latent_hidden = nn.Linear(latent_dim, hidden_dims[0])
        self.hidden_hidden = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.hidden_out  = nn.Linear(hidden_dims[1], out_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        h = self.latent_hidden(z)
        h = self.relu(h)
        h = self.hidden_hidden(h)
        h = self.relu(h)
        out = self.sigmoid(self.hidden_out(h))
        return out
    
class VAE(nn.Module):
    def __init__(self, in_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()

        self.encoder = Encoder(in_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim[::-1], in_dim)

    def rep_trick(self, mu, logvar):
        epsilon = torch.randn_like(logvar)
        z = mu + torch.sqrt(logvar.exp())*epsilon
        return z
    
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.rep_trick(mu, logvar)
        out = self.decoder(z)
        return out, mu, logvar