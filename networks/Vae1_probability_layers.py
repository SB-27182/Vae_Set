import torch
from torch import nn

#--------[Latent Density Used For Vae1]---------#
from torch.distributions import MultivariateNormal



class Probability_layer_gaussian(nn.Module):
    #--------[Constructor]---------#
    def __init__(self, input_dims, output_dims):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.mu = nn.Linear(input_dims, output_dims)
        self.log_std = nn.Linear(input_dims, output_dims)


    def reparameterized_sample(self, mu, variance_tensor):
        n = mu.shape[0]
        j = mu.shape[1]
        variance_tensor = variance_tensor.view(n, j, -1)
        #manual_identity_matrix = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        manual_identity_matrix = torch.eye(self.output_dims)
        covariance_matrix = manual_identity_matrix * variance_tensor

        latent_distribution = MultivariateNormal(loc=mu, covariance_matrix=covariance_matrix)
        z = latent_distribution.rsample()
        return z


    def forward(self, x):
        mean = self.mu(x)

        log_std = self.log_std(x)
        log_var = 2 * log_std
        variance = torch.exp(log_var)

        z = self.reparameterized_sample(mean, variance)
        return mean, variance, z




class Probability_layer_sigmoid_for_cBernoulli_likelihood(nn.Module):
    """
      NOTE: We are using MNIST data as our dataset.
             Thus, we use a sigmoid function in the final layer.

      NOTE: This is the reconstruction of x, thus re-parameterized sampling is not necessary.
      """
    #--------[Constructor]---------#
    def __init__(self, input_dims, output_dims):
        super().__init__()
        self.independence_layer = nn.Linear(input_dims, output_dims)
        self.k = nn.Sigmoid()

    def forward(self, z):
        z = self.independence_layer(z)
        k = self.k(z)
        return k