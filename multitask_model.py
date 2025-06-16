import geomloss
import torch
import torch.nn as nn
from torch.autograd import Function
import zuko
import matplotlib.pyplot as plt
import time

# Define device where the model will be trained
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def mmd_loss(x, y, sigma=1.0):
    xx = torch.mm(x, x.t())
    yy = torch.mm(y, y.t())
    xy = torch.mm(x, y.t())

    X_sq = xx.diag().unsqueeze(1) + xx.diag().unsqueeze(0) - 2 * xx
    Y_sq = yy.diag().unsqueeze(1) + yy.diag().unsqueeze(0) - 2 * yy
    XY_sq = x.pow(2).sum(dim=1).unsqueeze(1) + y.pow(2).sum(dim=1).unsqueeze(0) - 2 * xy

    X_exp = torch.exp(-X_sq / (2 * sigma**2))
    Y_exp = torch.exp(-Y_sq / (2 * sigma**2))
    XY_exp = torch.exp(-XY_sq / (2 * sigma**2))

    loss = (X_exp.mean() + Y_exp.mean() - 2 * XY_exp.mean()) * 0.5
    return loss

# Define encoder component
class Encoder(nn.Module):

    def __init__(self, 
                 input_dim: int, 
                 latent_dim=50,
                 n_layers=2,
                 dropout=0.5):
        
        super(Encoder, self).__init__()

        layers = []

        for _ in range(n_layers):

            # Start with batch normalization, then linear layer and ReLU activation
            layers.append(nn.BatchNorm1d(input_dim if _ == 0 else latent_dim))

            linear_layer = nn.Linear(input_dim if _ == 0 else latent_dim, latent_dim)
            # Initialize weights and biases with Xavier normal distribution
            nn.init.xavier_normal_(linear_layer.weight)
            if linear_layer.bias is not None:
                nn.init.zeros_(linear_layer.bias)
            layers.append(linear_layer)

            # Add dropout layer with 50% probability
            layers.append(nn.Dropout(p=dropout))

            # Add ReLU activation function
            layers.append(nn.SiLU())

        self.encoder = nn.Sequential(*layers)
        
    
    def forward(self, x):
        return self.encoder(x)


def kl_divergence(p, q):
    epsilon = 1e-6
    p = torch.clamp(p, min=epsilon)
    q = torch.clamp(q, min=epsilon)
    return torch.sum(p * torch.log(p / q), dim=-1)

def jensen_shannon_divergence(p, q):
    m = 0.5 * (p + q)
    jsd = 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)
    return jsd

def jensen_shannon_distance(p, q):
    jsd = jensen_shannon_divergence(p, q)
    jsd = torch.clamp(jsd, min=0.0)
    return torch.sqrt(jsd)

def sinkhorn_loss(
    x,
    y,
    blur,
):
    loss = geomloss.SamplesLoss("sinkhorn", blur=blur, scaling=0.9, reach=None)
    return loss(x, y)



class MultiTask(nn.Module):

    def __init__(self,
                 theta_dim: int, # Dimension of the probability distribution approximated by the flow
                 n_conditions: int, 
                 n_layers_enc = 2,
                 latent_dim_enc = 100,
                 n_transforms = 5,
                 n_layers_per_transform = 2,
                 n_neurons_flow = 50):

        super(MultiTask, self).__init__()

        # Define model components
        self.encoder = Encoder(input_dim=n_conditions,
                               n_layers=n_layers_enc,
                               latent_dim=latent_dim_enc)
        
        self.flow = zuko.flows.MAF(features=theta_dim,
                                   context=latent_dim_enc,
                                   transforms=n_transforms,
                                   hidden_features=[n_neurons_flow for i in range(n_layers_per_transform)])
        
        # Add loss function tunable weights
        self.eta_1 = nn.Parameter(torch.tensor(1.0, device=device))
        self.eta_2 = nn.Parameter(torch.tensor(1.0, device=device))
        self.eta_3 = nn.Parameter(torch.tensor(1.0, device=device)) #MMD

        # Add list for storing distances and eta values
        self.max_distances, self.js_distances = [], []
        self.blur_vals, self.eta_1_vals, self.eta_2_vals = [], [], []

        

    def forward(self, x):

        # Encode input into latent space
        x = self.encoder(x)

        return x
    
    def sample(self, 
               conditions: torch.Tensor, 
               n_samples: int):

        # Get encoded representation of the conditions
        x = self.encoder(conditions)

        # Sample from the amortised posterior
        samples = self.flow(x).sample((n_samples,))

        return samples
    
    def train_step(self, 
                   data, 
                   conditions,
                   validate=False,
                   warmup=False):    

        # Forward pass through the model
        encoded_data = self.encoder(data)

        # Calculate log-probability of data from the source domain
        idx_source = torch.isfinite(conditions).any(axis=1)
        idx_target = torch.logical_not(idx_source)

        source_features = encoded_data[idx_source]
        target_features = encoded_data[idx_target]

        log_p = -self.flow(source_features).log_prob(conditions[idx_source]).mean()

        if warmup:
            return log_p.item(), 0.0, log_p.item()

        # Calculate DA loss value
        # Ensures there are the same number of examples from the source and target domain
        n_samples = min(source_features.shape[0], target_features.shape[0])
    
        pairwise_distances = torch.cdist(source_features[:n_samples], target_features[:n_samples], p=2)
        flattened_distances = pairwise_distances.view(-1)
        max_distance = torch.max(flattened_distances)

        dynamic_blur_val = 0.05 * max_distance.detach().cpu().numpy()

        DA_loss = sinkhorn_loss(
            source_features[:n_samples],
            target_features[:n_samples],
            blur=max(dynamic_blur_val, 0.01),  # Apply lower bound to blur
        )

        MMD_loss = mmd_loss(source_features[:n_samples], 
                            target_features[:n_samples], sigma=1.0)

        loss = (
            (1 / (2 * self.eta_1**2)) * log_p
            + (1 / (2 * self.eta_2**2)) * DA_loss
            + MMD_loss / (2 * self.eta_3**2)
            + torch.log(torch.abs(self.eta_1) * torch.abs(self.eta_2)) * torch.abs(self.eta_3)
        )

        if validate:
            return loss.item(), DA_loss.item(), log_p.item()

        # Store distances and eta values
        self.max_distances.append(max_distance.item())
        self.js_distances.append(
            jensen_shannon_distance(source_features[:n_samples], target_features[:n_samples])
            .nanmean()
            .item()
        )
        self.blur_vals.append(dynamic_blur_val)
        self.eta_1_vals.append(self.eta_1.item())
        self.eta_2_vals.append(self.eta_2.item())

        # Backpropagation
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=10.0)
        self.eta_1.data.clamp_(min=1e-3)
        self.eta_2.data.clamp_(min=0.25 * self.eta_1.data.item())
        self.eta_3.data.clamp_(min=0.25 * self.eta_1.data.item())        

        return loss.item(), DA_loss.item(), log_p.item()
    
    
    def train_model(self, 
                train_dataloader: torch.utils.data.DataLoader,
                val_dataloader: torch.utils.data.DataLoader,
                optimizer: torch.optim.Optimizer,
                epochs: int,
                n_warmup_epochs: int = 5
                ):
        
        #optimizer.add_param_group({"params": [self.eta_1, self.eta_2]})

        # Initialize loss function list
        train_losses, train_flow_losses, train_DA_losses = [], [], []
        val_losses, val_flow_losses, val_DA_losses = [], [], []


        for epoch in range(epochs):

            warmup = True if epoch < n_warmup_epochs else False

            train_loss = 0.0
            DA_loss_epoch = 0.0
            log_p_epoch = 0.0

            val_loss = 0.0
            val_DA_loss_epoch = 0.0
            val_log_p_epoch = 0.0

            start_time = time.time()
            
            for (train_data,train_conditions), (val_data,val_conditions) in zip(train_dataloader, train_dataloader):

                # Set model to training mode
                self.train()
                optimizer.zero_grad()
                # Train step
                loss, DA_loss, log_p = self.train_step(train_data, 
                                                       train_conditions,
                                                       warmup=warmup)
                # Update model weights
                optimizer.step()

                ################################################################

                # Validation step
                self.eval()
                with torch.no_grad():
                    val_loss, val_DA_loss, val_log_p = self.train_step(val_data, 
                                                                       val_conditions, 
                                                                       validate=True)

                # Accumulate losses
                train_loss += loss
                DA_loss_epoch += DA_loss
                log_p_epoch += log_p

                val_loss += val_loss
                val_DA_loss_epoch += val_DA_loss
                val_log_p_epoch += val_log_p


                ################################################################
                

            # Calculate average losses for the epoch
            train_loss /= len(train_dataloader)
            DA_loss_epoch /= len(train_dataloader)
            log_p_epoch /= len(train_dataloader)

            val_loss /= len(val_dataloader)
            val_DA_loss_epoch /= len(val_dataloader)
            val_log_p_epoch /= len(val_dataloader)

            train_losses.append(train_loss)
            train_flow_losses.append(log_p_epoch)
            train_DA_losses.append(DA_loss_epoch)

            val_losses.append(val_loss)
            val_flow_losses.append(val_log_p_epoch)
            val_DA_losses.append(val_DA_loss_epoch)

            end_time = time.time()

            #############################################################

            print(
                f"Epoch [{epoch+1}/{epochs}] | "
                f"Train Loss: Total={train_loss:.4f}, DA={DA_loss_epoch:.4f}, Flow={log_p_epoch:.4f} | "
                f"Val Loss: Total={val_loss:.4f}, DA={val_DA_loss_epoch:.4f}, Flow={val_log_p_epoch:.4f} | "
                f"Time: {end_time - start_time:.2f}s",
                flush=True
            )

        training_results = {
            "train_loss": train_losses,
            "train_flow_loss": train_flow_losses,
            "train_DA_loss": train_DA_losses,
            "val_loss": val_losses,
            "val_flow_loss": val_flow_losses,
            "val_DA_loss": val_DA_losses,
            "max_distances": self.max_distances,
            "js_distances": self.js_distances
        }

        return training_results
    

def plot_training_losses(training_results):

    import numpy as np

    # Unpack the training results
    epochs_loss = training_results["train_loss"]
    epochs_flow_loss = training_results["train_flow_loss"]
    epochs_DA_loss = training_results["train_DA_loss"]
    epochs_val_loss = training_results["val_loss"]
    epochs_val_flow_loss = training_results["val_flow_loss"]    
    epochs_val_DA_loss = training_results["val_DA_loss"]
    
    # Plot the training losses
    epochs = [i for i in range(1, len(epochs_loss) + 1)]
    fig,ax = plt.subplots(1,1,figsize=(10,5))

    #  Normalize the losses
    epochs_loss = [loss / np.median(epochs_loss) for loss in epochs_loss]
    epochs_flow_loss = [loss / np.median(epochs_flow_loss) for loss in epochs_flow_loss]
    epochs_DA_loss = [loss / np.median(epochs_DA_loss) for loss in epochs_DA_loss]
    epochs_val_loss = [loss / np.median(epochs_val_loss) for loss in epochs_val_loss]
    epochs_val_flow_loss = [loss / np.median(epochs_val_flow_loss) for loss in epochs_val_flow_loss]
    epochs_val_DA_loss = [loss / np.median(epochs_val_DA_loss) for loss in epochs_val_DA_loss]

    ax.plot(epochs, epochs_loss, label='Total Loss', ls=':', color='k')
    ax.plot(epochs, epochs_DA_loss, label='Classifier Loss', ls=':', color='r')
    ax.plot(epochs, epochs_flow_loss, label='Decoder Loss', ls=':', color='b')
    ax.plot(epochs, epochs_val_loss, label='Total Loss', ls='-', color='k')
    ax.plot(epochs, epochs_val_DA_loss, label='Classifier Loss', ls='-', color='r')
    ax.plot(epochs, epochs_val_flow_loss, label='Decoder Loss', ls='-', color='b')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Normalised Loss values')
    ax.legend(loc='upper right')

    ax.set_ylim([-1,10])
    
    return fig

def plot_distances(training_results):

    # Unpack the training results
    max_distances = training_results["max_distances"]
    js_distances = training_results["js_distances"]
    
    # Plot the distances and eta values
    fig, ax = plt.subplots(2, 1, figsize=(10, 15))

    ax[0].plot(max_distances, label='Max Distance')
    ax[0].set_xlabel('Steps')
    ax[0].set_ylabel('Max Distance')
    ax[0].legend(loc='upper right')

    ax[1].plot(js_distances, label='Jensen-Shannon Distance')
    ax[1].set_xlabel('Steps')
    ax[1].set_ylabel('Jensen-Shannon Distance')
    ax[1].legend(loc='upper right')

    return fig
                





