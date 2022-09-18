import torch
from torch import nn

#----The Probability Layers of This Network----#
from networks.Vae1_probability_layers import *



class Vae1_network(nn.Module):
    #--------[Constructor]---------#
    def __init__(self, x_dim, z_dim):
        super().__init__()
        self.encoder_network = EncoderNet(x_dim, z_dim)
        self.decoder_network = DecoderNet(x_dim, z_dim)

    def forward(self, x):
        assert x.dim() == 2, "x-input was not a 2-dim flattened tensor!"
        latent_mapped_data = self.encoder_network(x)
        z = latent_mapped_data.get("z_x")

        decoder_mapped_data = self.decoder_network(z)
        return latent_mapped_data|decoder_mapped_data



class EncoderNet(nn.Module):
    #--------[Constructor]---------#
    def __init__(self, x_dim, z_dim):
        super().__init__()
        # q(z|x)
        self.qz_x_network = torch.nn.ModuleList([
            nn.Linear(x_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            Probability_layer_gaussian(512, z_dim)
        ])

    # q(z|x)
    def qz_x_call(self, x):
        for layer in self.qz_x_network:
            x = layer(x)
        z = x
        return(z)


    def forward(self, x):
        muZ_x, varZ_x, z_x = self.qz_x_call(x)
        return {
            "muZ_x": muZ_x, "varZ_x": varZ_x, "z_x": z_x
        }



class DecoderNet(nn.Module):
    #--------[Constructor]---------#
    def __init__(self, x_dim, z_dim):
        super().__init__()
        # p(x|z)
        self.px_z_network = torch.nn.ModuleList([
            nn.Linear(z_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            Probability_layer_sigmoid_for_cBernoulli_likelihood(512, x_dim)
        ])

    # f(x|z)
    def fx_z_call(self, z):
        for layer in self.px_z_network:
            z = layer(z)
        x = z
        return(x)


    def forward(self, z):
        sigmoid_x = self.fx_z_call(z)
        return {
            "sigmoid_x": sigmoid_x
        }

