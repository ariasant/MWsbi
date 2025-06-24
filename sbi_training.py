import ili
from ili.inference import InferenceRunner
from ili.dataloaders import NumpyLoader
from ili.validation.metrics import PosteriorCoverage
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pickle
import torch
# ignore warnings for readability
import warnings
warnings.filterwarnings('ignore')


# Run on gpu if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device:', device, flush=True)





def NPE_training(X_train, 
                 Y_train, 
                 filename,
                 prior_ranges=None,
                 plot_dir = '/mnt/aridata1/users/ariasant/auriga-sbi/plots/with_satellites/',
                 output_dir = '/mnt/aridata1/users/ariasant/auriga-sbi/posteriors/100+/with_satellites/'):

    # Make dataloader object
    loader = NumpyLoader(x=X_train, 
                         theta=Y_train)


    #################################################################################################
    #################################################################################################
    # INFERENCE
    #################################################################################################
    #################################################################################################

    n_nets = 3
    n_nodes_per_net = 200
    n_transforms = 10
    learning_rate = 0.0001
    batch_size = 256

    # Define prior distributions for accretion redshift and progenitor mass
    if prior_ranges is None:
        prior = ili.utils.Uniform(low=[Y_train[:,i].min() for i in range(Y_train.shape[1])],
                                  high=[Y_train[:,i].max() for i in range(Y_train.shape[1])], 
                                  device=device)
        
    else:
        prior = ili.utils.Uniform(low=prior_ranges[0],
                                  high=prior_ranges[1], 
                                  device=device)



    # instantiate neural networks to be used as an ensemble
    nets = [ili.utils.load_nde_sbi(engine='NPE', 
                                   model='maf', 
                                   hidden_features=n_nodes_per_net, 
                                   num_transforms=n_transforms)
            for n in range(n_nets)]

    # define training arguments
    train_args = {
        'training_batch_size': batch_size,
        'learning_rate': learning_rate
    }

    # initialize the trainer
    runner = InferenceRunner.load(
        backend='sbi',
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
    ax.set_ylim([-10,10])
    ax.legend()
    fig.savefig(output_dir+f'{filename}_training_plot.png', dpi=400)

    # Save posterior
    pickle.dump(posterior_ensemble, 
                open(f'{output_dir}{filename}.pkl', 'wb'))

    return posterior_ensemble


def validation(posterior_ensemble, 
               test_dictionary, 
               filename,
               output_dir='/mnt/aridata1/users/ariasant/auriga-sbi/samples/100+/with_satellites/'):

    # Generate samples from posterior for test examples
    seed_samp = 42
    torch.manual_seed(seed_samp)

    test_samples = {}
    for X_obs, theta_fid, progID in zip(test_dictionary['X'],
                                        test_dictionary['Y'],
                                        test_dictionary['ID']):

        samples = posterior_ensemble.sample((1000,),
                                            torch.Tensor(X_obs).to(device)
                                            )
        # calculate the log_prob for each sample
        log_prob = posterior_ensemble.log_prob(samples, torch.Tensor(X_obs).to(device), norm_posterior=False)

        test_samples[progID] = (samples.cpu().numpy(), 
                                log_prob.cpu().numpy(),
                                theta_fid)

    
    metric = PosteriorCoverage(
        num_samples=1000, sample_method='direct', 
        labels=[f'$\\theta_{i}$' for i in range(len(theta_fid))],
        plot_list = ["coverage", "tarp", "logprob"],
        out_dir=None
    )

    try:
        figs = metric(
            posterior=posterior_ensemble, 
            x=test_dictionary['X'], theta=test_dictionary['Y']
        )

        for i,fig in enumerate(figs):
            fig.savefig(f"{output_dir}{filename}_validation{i+1}.png", dpi=400)

    except:
        pass
    
    return test_samples









