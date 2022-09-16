import torch

#-----------For CSV output-------------#
import numpy as np
import pandas as pd


#-------The Analysis-type Model--------#
from models_with_analysis.Vae1_with_analysis import *



#=================================# Args #=================================#
arguments = {
    "save_file": "./meta_data/trainedVae1_0.pt",

    "latent_dimensions": 2,
    "input_dimensions": 3,

    "csv_file":"./meta_data/csvOut_Vae1_0.csv"
}

#=============================# Ana-Model #================================#
ana_model = Vae1_with_analysis(args=arguments)


#===========================# Generate CSV #===============================#
observations = 100
test_val = torch.tensor([[0.5,0.5,0.5]])
sample = ana_model.latent_distribution_1(x=test_val, n=observations)
sample_np = sample.numpy()
sample_df = pd.DataFrame(sample_np)
sample_df.to_csv(arguments.get("csv_file"))
