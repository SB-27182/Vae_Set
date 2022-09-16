import torch
import matplotlib.pyplot as plt


#------------The Single Network Used By Vae1---------------#
from networks.Vae1_networks import *


class Vae1_with_analysis:
    #--------[Constructor]---------#
    def __init__(self, args):
        self.latent_dims = args.get("latent_dimensions")
        self.input_dims = args.get("input_dimensions")
        self.network = Vae1_network(x_dim=self.input_dims, z_dim=self.latent_dims)

        self.save_file = args.get("save_file")
        checkpoint = torch.load(self.save_file)
        self.network.load_state_dict(checkpoint.get("model_state_dict"))

        self.enc = self.network.encoder_network
        self.dec = self.network.decoder_network



    def __manually_translated_tensor_xSigmoid(self, z_mu, z_shift_direction, scale_each_shift, n_observations):
        """Helper method:
           Input: z tensor  (size:[1, latent_dims])
           Output: one x tensors (size:[1, input_dims])

           Usage: With a mean and variance value, iteratively shift out in the variance direction"""
        tlist = []
        i = 1
        while(i <= n_observations):
            new_tensor = z_mu + z_shift_direction * scale_each_shift * i
            xSig = self.dec.fx_z_call(new_tensor)
            tlist.append(xSig.numpy())
            i = i + 1
        return tlist



    def __parameter_translated_tensor_zGaussian(self, x, var_difference_scalar, n_observations):
        """Helper method:
           Input: x tensor  (size:[1, input_dims])
           Output: the mapped parameters (mu, var) are
                   used to generate n (var*unit)-shifted observations of z ."""
        z_dict = self.enc.qz_x_call(x)
        var = z_dict.get("varZ_x")
        mu = z_dict.get("muZ_x")
        tlist = []
        i = 1
        while(i <= n_observations):
            ith_unit_difference = var_difference_scalar * i
            new_tensor = mu + var * ith_unit_difference
            tlist = tlist.append(new_tensor)
            i = i + 1
        return tlist



    def observations_one_side_symmetric_latent(self, x, n):
        assert x.shape[0] == 1, "This method requires only one x input!"
        with torch.no_grad():
            mu, var, z = self.enc.qz_x_call(x)
            std = torch.sqrt(var)

            tList_positive = self.__manually_translated_tensor_xSigmoid(z_mu=mu, z_shift_direction=std, scale_each_shift=0.5, n_observations=25)
        return tList_positive



    def latent_distribution_1(self, x, n): #FIXME!! Too many shortcuts. Not explicit enough.
        """For a given x, parameterizes a latent distribution, and generates n observations of z.
            Maps the z tensor to n observations of x."""
        self.network.eval()
        with torch.no_grad():
            mu, var, z = self.enc.qz_x_call(x)
            cov = torch.eye(var.shape[1]) * var
            dist = torch.distributions.MultivariateNormal(mu, cov)
            sample = dist.sample([n])
        return self.dec.fx_z_call(sample)



