import torch

#--------This Model's Network--------#
from networks.Vae1_networks import *

#-----This Model's Loss Functions----#
from costFunctions.Vae1_likelihoodFunctions import *

#---This Model's Optimization Alg.---#
from torch.optim import Adam



class Vae1:
    #------[Constructor]-------#
    def __init__(self, args):
        self.latent_dims = args.get("latent_dimensions")
        self.input_dims = args.get("input_dimensions")
        self.beta_temp = args.get("beta_temp")
        self.network = Vae1_network(x_dim=self.input_dims, z_dim=self.latent_dims)

        self.batch_size = args.get("batch_size")
        self.gradient_jump = args.get("learning_rate")
        self.num_epochs = args.get("number_of_epochs")

        self.report_batch = args.get("reporter_batches")
        self.reporter_file = args.get("report_file")


    def train(self, data_loader):
        my_optimizer = Adam(self.network.parameters(), lr=self.gradient_jump)
        #--[Here we point the optimizer at ALL of Vae1's parameters.]
        #--[Because it's called outside a train_epoch, the optimization carries over for the duration of the whole "for" loop below. That is, each epoch.]

        for e in range(1, self.num_epochs+1):
            this_epoch = self.train_epoch(my_optimizer, data_loader)
            print("For epoch:", str(e), " , batch-loss is: ", str(this_epoch.get("epochs_loss")))



    def train_epoch(self, my_optimizer, data_loader):
        self.network.train()
        total_epoch_loss = 0.0

        for (i, data) in enumerate(data_loader):
            my_optimizer.zero_grad()
            input_data = data.view(data.size(0), -1)


            batchs_outputs_of_network = self.network(input_data)
            batchs_loss_dict = self.calculate_loss(input_data, batchs_outputs_of_network)
            batchs_loss_value = batchs_loss_dict["batchs_loss_value"]

            batchs_loss_value.backward()
            my_optimizer.step()

            total_epoch_loss = batchs_loss_value.item() + total_epoch_loss


        this_epochs_report = {
            "epochs_loss": total_epoch_loss,
            "batch_normalized_epochs_loss": total_epoch_loss / self.batch_size
        }

        return this_epochs_report



    def calculate_loss(self, input_data, batchs_outputs_of_network):
        sigmoid_x = batchs_outputs_of_network.get("sigmoid_x")
        muZ_x = batchs_outputs_of_network.get("muZ_x")
        varZ_x = batchs_outputs_of_network.get("varZ_x")
        z_x = batchs_outputs_of_network.get("z_x")

        qz_x = logLikelihood_gaussian(mu=muZ_x, variance=varZ_x, resampled_data=z_x)
        fz = logLikelihood_gaussian_standard(resampled_data=z_x)
        fx_z = logLikelihood_cBernoulli(k=sigmoid_x, x=input_data)

        inv_ln_mut_in = qz_x - fx_z - fz

        batchs_loss = torch.sum(inv_ln_mut_in, dim=0)
        return {
            "batchs_loss_value": batchs_loss
        }
