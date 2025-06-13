import ili
from ili.inference import InferenceRunner
from ili.dataloaders import NumpyLoader
from ili.validation.metrics import PosteriorCoverage
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pickle
import torch
from torch_geometric import nn as gnn
from torch_geometric.nn import aggr
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from ili.dataloaders import TorchLoader
# ignore warnings for readability
import warnings
warnings.filterwarnings('ignore')

# Run on gpu if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device:', device, flush=True)

# Design a custom Graph Attention Network embedder
class GATNetwork(nn.Module):
    def __init__(
        self, in_channels, gcn_channels, gcn_heads,
        dense_channels, out_channels, drop_p=0.1,
        edge_dim=None
    ):
        super().__init__()
        self.in_channels = in_channels
        self.dense_channels = dense_channels
        self.out_channels = out_channels
        self.drop_p = drop_p
        self.edge_dim = edge_dim
        self.gcn_channels = gcn_channels
        self.gcn_heads = gcn_heads

        self.graph_aggr = aggr.MultiAggregation(
            aggrs=['sum', 'mean', 'std', 'min',
                   'max', aggr.SoftmaxAggregation(learn=True)],
            mode='cat'
        )
        self.dropout = torch.nn.Dropout(p=self.drop_p)

        self._build_gnn()
        self._build_dnn(gcn_channels[-1]*len(self.graph_aggr.aggrs))

    def _build_dnn(self, in_channels):
        self.fc1 = torch.nn.Linear(in_channels, self.dense_channels[0])
        self.fcs = torch.nn.ModuleList(
            [torch.nn.Linear(self.dense_channels[i], self.dense_channels[i+1])
             for i in range(0, len(self.dense_channels)-1)])
        self.fc2 = torch.nn.Linear(self.dense_channels[-1], self.out_channels)

    def dnn(self, x):
        x = F.relu(self.fc1(x))
        for fc in self.fcs:
            x = self.dropout(x)
            x = F.relu(fc(x))
        x = self.fc2(x)
        return x

    def _build_gnn(self):
        self.conv1 = gnn.GATv2Conv(
            self.in_channels, self.gcn_channels[0],
            heads=self.gcn_heads[0], edge_dim=self.edge_dim)
        self.convs = torch.nn.ModuleList(
            [gnn.GATv2Conv(self.gcn_channels[i]*self.gcn_heads[i],
                           self.gcn_channels[i+1], heads=self.gcn_heads[i+1],
                           edge_dim=self.edge_dim)
             for i in range(len(self.gcn_channels)-2)]
        )
        self.conv2 = gnn.GATv2Conv(
            self.gcn_channels[-2]*self.gcn_heads[-2],
            self.gcn_channels[-1], heads=self.gcn_heads[-1],
            concat=False, edge_dim=self.edge_dim)

    def gnn(self, x, edge_index, edge_attr):
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        for conv in self.convs:
            x = F.relu(conv(x, edge_index, edge_attr))
        x = self.conv2(x, edge_index, edge_attr)
        return x

    def forward(self, x):
        node_features = torch.ones(x.num_nodes, 1).to(device)
        edge_index, edge_attr = x.edge_index, x.edge_attr
        ptr = x.ptr if hasattr(x, 'ptr') else None

        x = self.gnn(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
        x = self.graph_aggr(x, ptr=ptr)
        x = self.dnn(x)
        return x


def NPE_training(train_data,
                 val_data,
                 collate_fn, 
                 prior_ranges: tuple,
                 filename: str,
                 batch_size: int=32,
                 output_dir = '/mnt/aridata1/users/ariasant/auriga-sbi/posteriors/100+/with_satellites/'):


    batch_size = 32

    # Make dataloader object
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    loader = TorchLoader(train_loader, val_loader)
    
    # Initialise embedding network
    embedding = GATNetwork(
                in_channels=1, gcn_channels=[4, 8],
                gcn_heads=[4, 4],
                dense_channels=[32, 16], out_channels=6,
                edge_dim=1
    )


    #################################################################################################
    #################################################################################################
    # INFERENCE
    #################################################################################################
    #################################################################################################

    n_nets = 3
    n_nodes_per_net = 463
    n_transforms = 22
    learning_rate = 0.001
    batch_size = 2569

    # Define prior distributions for accretion redshift and progenitor mass
    prior = ili.utils.Uniform(low=prior_ranges[0],
                              high=prior_ranges[1], 
                              device=device)

    # instantiate neural networks to be used as an ensemble
    nets = [ili.utils.load_nde_sbi(engine='NPE', 
                                   model='maf', 
                                   hidden_features=n_nodes_per_net, 
                                   num_transforms=n_transforms,
                                   embedding_net=embedding)
            for n in range(n_nets)]

    # define training arguments
    train_args = {
        'training_batch_size': batch_size,
        'learning_rate': learning_rate
    }

    # initialize the trainer
    runner = InferenceRunner.load(
        backend='lampe',
        engine='NPE',
        prior=prior,
        nets=nets,
        device=device,
        train_args=train_args,
    )

    # Train the model
    posterior_ensemble, summaries = runner(loader=loader)


    # Plot train/validation loss
    fig, ax = plt.subplots(1, 1, figsize=(6,4))
    c = list(mcolors.TABLEAU_COLORS)
    for i, m in enumerate(summaries):
        ax.plot(m['training_log_probs'], ls='-', label=f"{i}_train", c=c[i])
        ax.plot(m['validation_log_probs'], ls='--', label=f"{i}_val", c=c[i])
    ax.set_xlim(0)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Log probability')
    ax.set_ylim([-10,5])
    ax.legend()
    fig.savefig(output_dir+f'{filename}_training_plot.png', dpi=400)

    # Save posterior
    pickle.dump(posterior_ensemble, 
                open(f'{output_dir}{filename}.pkl', 'wb'))

    return posterior_ensemble


def validation(posterior_ensemble, 
               val_data, 
               filename,
               output_dir='/mnt/aridata1/users/ariasant/auriga-sbi/samples/100+/with_satellites/'):

    # Generate samples from posterior for test examples
    seed_samp = 42
    torch.manual_seed(seed_samp)

    test_samples = {}
    for data in zip(val_data):

        samples = posterior_ensemble.sample((1000,),
                                            data
                                            )
        # calculate the log_prob for each sample
        log_prob = posterior_ensemble.log_prob(samples, data)

        test_samples[data.progID] = (samples.cpu().numpy(), 
                                    log_prob.cpu().numpy(),
                                    data.y.cpu().numpy())

    
    metric = PosteriorCoverage(
        num_samples=1000, sample_method='direct', 
        labels=[f'$\\theta_{i}$' for i in range(len(data.y.cpu().numpy()))],
        plot_list = ["coverage", "tarp", "logprob"],
        out_dir=None
    )

    try:
        figs = metric(
            posterior=posterior_ensemble, 
            x=data, theta=torch.vstack([item.y for item in data])
        )

        for i,fig in enumerate(figs):
            fig.savefig(f"{output_dir}{filename}_validation{i+1}.png", dpi=400)

    except:
        pass
    
    return test_samples









