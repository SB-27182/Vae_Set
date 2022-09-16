import torch
import numpy as np

#--------For Reproducibility--------#
import random

#----------For MNIST Data-----------#
from torchvision import datasets, transforms

#-------------Utilities-------------#
from torch.utils.data import DataLoader, Dataset

#----We are training THIS model-----#
from models.Vae1_with_reporter import *



#=============================# Save File #================================#
SAVE_FILE = "./meta_data/trainedVae1_0.pt"


#================================# Seed #==================================#
SEED = 15
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


#================================# Args #==================================#
arguments = {
    "latent_dimensions": 2,
    "input_dimensions": 3,
    "beta_temp": 0.95,

    "batch_size": 100,
    "learning_rate": 0.00001,
    "number_of_epochs": 200,

    "report_file": './meta_data/reported_batches_vae1.txt',
    "reporter_batches": [0, 20]

}


#================================# Data #==================================#
class torch_to_torch_data(Dataset):
    def __init__(self):
        data = torch.load("./data/5k_images.pt")
        self.data = data
        self.n__samples = data.shape[0]

    def __getitem__(self, index):
        return self.data[index, :]

    def __len__(self):
        return self.n__samples

class csv_to_torch_data(Dataset):
    def __init__(self):
        tokens = np.loadtxt("./training_data/norm_0.csv", delimiter=',', dtype=np.float32)
        self.data = torch.from_numpy(tokens)
        self.n_samples = tokens.shape[0]

    def __getitem__(self, index):
        return self.data[index, :]

    def __len__(self):
        return self.n_samples

#dataset = datasets.MNIST('./training_data/mnist', train=True, download=True, transform=transforms.ToTensor())
dataset = csv_to_torch_data()
train_loader = DataLoader(dataset=dataset, batch_size=arguments.get("batch_size"), shuffle=True)


#================================# Model #=================================#
model = Vae1(args=arguments)


#================================# Train #=================================#
model.train(train_loader)


#================================# Save #==================================#
print("Saving to: ", SAVE_FILE)
torch.save({'model_state_dict': model.network.state_dict()}, SAVE_FILE)