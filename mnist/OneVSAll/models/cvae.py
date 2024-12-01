import torch
import torch.nn as nn

class Encoder(nn.Module):

    def __init__(self, in_channels, hidden_channels=64, z_dim=10):
        super(Encoder, self).__init__()
        self.disc = nn.Sequential(
            self.make_disc_block(in_channels, hidden_channels),
            self.make_disc_block(hidden_channels, hidden_channels * 2),
            self.make_disc_block(hidden_channels * 2, 2*z_dim, final_layer=True),
        )

    def make_disc_block(self, input_channels, output_channels, kernel_size=4, stride=2, final_layer=False):

        if not final_layer:
            return nn.Sequential(
                nn.Conv2d(in_channels=input_channels, out_channels=output_channels,
                          kernel_size=kernel_size, stride=stride),
                nn.BatchNorm2d(output_channels),
                nn.LeakyReLU(negative_slope=0.2)
            )
        else:  
            return nn.Sequential(
                nn.Conv2d(in_channels=input_channels, out_channels=output_channels,
                          kernel_size=kernel_size, stride=stride)
            )

    def forward(self, image):
        disc_pred = self.disc(image)
        return disc_pred.view(len(disc_pred), -1).chunk(2, dim=1)
    
class Decoder(nn.Module):

    def __init__(self, z_dim=10, hidden_channels=64, out_channels=1):
        super(Decoder, self).__init__()
        self.z_dim = z_dim
        self.gen = nn.Sequential(
            self.make_gen_block(z_dim, hidden_channels * 4),
            self.make_gen_block(hidden_channels * 4, hidden_channels *
                                2, kernel_size=4, stride=1),
            self.make_gen_block(hidden_channels * 2, hidden_channels),
            self.make_gen_block(hidden_channels, out_channels,
                                kernel_size=4, final_layer=True),
        )

    def make_gen_block(self, input_channels, output_channels, kernel_size=3, stride=2, final_layer=False):

        if not final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=input_channels, out_channels=output_channels, kernel_size=kernel_size, stride=stride),
                nn.BatchNorm2d(output_channels),
                nn.ReLU()
            )
        else:  # Final Layer
            return nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=input_channels, out_channels=output_channels, kernel_size=kernel_size, stride=stride),
                nn.Sigmoid()
            )

    def unsqueeze_noise(self, noise):
        return noise.view(len(noise), self.z_dim, 1, 1)

    def forward(self, noise):
        x = self.unsqueeze_noise(noise)
        return self.gen(x)
    

class CVAE(nn.Module):
    def __init__(self, in_channels, hidden_channels, latent_dim):
        super(CVAE, self).__init__()

        self.encoder = Encoder(in_channels, hidden_channels, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_channels, in_channels)

    def rep_trick(self, mu, logvar):
        epsilon = torch.randn_like(logvar)
        z = mu + torch.sqrt(logvar.exp())*epsilon
        return z
    
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.rep_trick(mu, logvar)
        out = self.decoder(z)
        return out, mu, logvar