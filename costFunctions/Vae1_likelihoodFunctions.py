import math
import torch



def logLikelihood_gaussian(mu, variance, resampled_data):
    assert torch.all(variance > 0.0), "Variance value was negative!"
    assert mu.shape[0] == resampled_data.shape[0] == variance.shape[0], "Batch sizes of gauss not same size!"
    assert mu.shape[1] == resampled_data.shape[1] == variance.shape[1], "Latent dimensions of gauss not same size!"
    j = mu.shape[1]
    n = mu.shape[0]
    #[Note: What if variance is 0.0?  ||  Because variance = exp(log_var) , the only way to get a zero variance is if the neural network outputs -infinite.]
    #[Experiment: Use a very small base like 0.01]

    #The Determinant:
    variance = variance.view(n, j, -1)
    #manual_identity_matrix = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    manual_identity_matrix = torch.eye(j)
    cov_mat = variance * manual_identity_matrix
    det = torch.linalg.det(cov_mat)
    lnDet = torch.log(det)

    #The Kernel:
    sub = resampled_data - mu
    distance = torch.linalg.solve(cov_mat, sub)   #[Note: Faster than computing the inverse of cov_mat; example: linalg.inv(cov_mat) @ sub  ==  linalg.solve(cov_mat, sub)  ]
    ratch_dot_prod = sub * distance
    distance = ratch_dot_prod.sum(dim=1)

    #Total Likelihood:
    logLikelihood = -j * math.log(2*math.pi) - lnDet - distance
    logLikelihood = logLikelihood * 0.5
    return logLikelihood


def logLikelihood_gaussian_standard(resampled_data):
    j = resampled_data.shape[1]
    n = resampled_data.shape[0]
    manual_identity_matrix = torch.eye(j)
    manual_identity_matrix = manual_identity_matrix.unsqueeze(0).repeat(n, 1, 1)

    manual_mean_vector = torch.zeros([n, j])

    det = torch.linalg.det(manual_identity_matrix)   #FIXME: This is overly pedantic. We know what the determinant of the identity-basis matrix explicitly!
    lnDet = torch.log(det)

    sub = resampled_data - manual_mean_vector
    distance = torch.linalg.solve(manual_identity_matrix, sub)
    ratchet_dot_prod = sub * distance
    distance = ratchet_dot_prod.sum(dim=1)

    logLikelihood = -j * math.log(2 * math.pi) - lnDet - distance
    logLikelihood = logLikelihood * 0.5
    return logLikelihood


def logLikelihood_cBernoulli(k, x):
    """NOTE: Our test data set is MNIST data.
             This is why we use cBernoulli as the reconstruction density (close-to-binaric)."""

    assert x.shape[0] == k.shape[0], "Batch sizes of bernoulli not all same!"
    assert x.shape[1] == k.shape[1], "Observation dimensions of bernoulli not all same!"
    assert torch.all(x < 1.0), "Data values where not all less than one!"
    assert torch.all(x > 0.0), "Data values where not all greater than zero!"
    assert torch.all(k < 1.0), "Sigmoid x values where not all less than one!"
    assert torch.all(k > 0.0), "Sigmoid x values where not all greater than zero!"
    x_compliment = 1 - x
    k_compliment = 1 - k
    k_c2 = 1 - 2 * k
    bern_and_ln2 = x * torch.log(k) + x_compliment * torch.log(k_compliment) + math.log(2)


    iHyperbTan_taylor_approximation =  k_c2 + 0.3333 * torch.pow(input=k_c2, exponent=3) + \
           0.2 * torch.pow(input=k_c2, exponent=5) + 0.142 * torch.pow(input=k_c2, exponent=7) + \
           0.1111 * torch.pow(input=k_c2, exponent=9) + 0.09 * torch.pow(input=k_c2, exponent=11) + \
           0.0769 * torch.pow(input=k_c2, exponent=13) + 0.0666 * torch.pow(input=k_c2, exponent=15)

    iHtan_taylor_positive_mapped = iHyperbTan_taylor_approximation / k_c2
    log_taylor_aprox = torch.log(iHtan_taylor_positive_mapped)


    log_likelihood = bern_and_ln2 + log_taylor_aprox
    sum_indp_dims = torch.sum(log_likelihood, dim=1)
    return sum_indp_dims





