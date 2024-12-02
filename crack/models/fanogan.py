import torch 
import torch.nn as nn 


class ConvGenerator(nn.Module):
    def __init__(self, z_dim, hidden_dim):
        super(ConvGenerator, self).__init__()

        self.gen = nn.Sequential(
            self.make_gen_block(z_dim, hidden_dim*4),
            self.make_gen_block(hidden_dim*4, hidden_dim*2, kernel_size=4, stride=1),
            self.make_gen_block(hidden_dim*2, hidden_dim),
            self.make_gen_block(hidden_dim, 1, kernel_size=4, final_layer=True)
        )

    def make_gen_block(self, in_channels, out_channels, kernel_size=3, stride=2,final_layer=False):
        if not final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride),
                nn.Sigmoid()
            ) 
        
    def unsqueeze_noise(self, noise):
        batch, z_dim = noise.size(0), noise.size(1)
        return noise.view(batch, z_dim, 1, 1)
    
    def forward(self, z):
        z = self.unsqueeze_noise(z)
        return self.gen(z)
    
    
class ConvDiscriminator(nn.Module):
    def __init__(self, im_channel, hidden_dim=16):
        super(ConvDiscriminator, self).__init__()

        self.disc = nn.Sequential(
            self.make_disc_block(im_channel, hidden_dim),
            self.make_disc_block(hidden_dim, hidden_dim*2),
            self.make_disc_block(hidden_dim*2, 1, final_layers=True)
        )

    def make_disc_block(self, in_channels, out_channels, kernel_size=4, stride=2, final_layers=False):
        if not final_layers:
            return nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2)
            )
        else:
            return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride)
        
    def forward(self, image):
        return torch.sigmoid(self.disc(image).view(-1, 1))

class Encoder(nn.Module):

    def __init__(self, in_channels, hidden_channels=64, z_dim=10):
        super(Encoder, self).__init__()
        self.disc = nn.Sequential(
            self.make_disc_block(in_channels, hidden_channels),
            self.make_disc_block(hidden_channels, hidden_channels * 2),
            self.make_disc_block(hidden_channels * 2, z_dim, final_layer=True),
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
        return disc_pred.view(len(disc_pred), -1)