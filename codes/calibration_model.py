import numpy as np
import pymc as pm
from imblearn.over_sampling import RandomOverSampler


# Set numpy random key
rng = np.random.default_rng(17)

# Generate noise realisation to mimic Milky Way observations

def generate_mean_cov_model(features, 
                            n_stars_per_prog):

    n_features = len(features)

    coords = {"features": features, 
              "features_bis": features, 
              "star_id": np.arange(n_stars_per_prog)}

    with pm.Model(coords=coords) as model:
        chol, corr, stds = pm.LKJCholeskyCov(
            "chol", 
            n=n_features, 
            eta=1, 
            sd_dist=pm.Exponential.dist(100, shape=n_features)
        )
        mu_components = pm.math.stack([pm.Normal("mu_E", mu=0.00, sigma=0.10),
                                       pm.Normal("mu_L", mu=0.00, sigma=0.10),
                                       pm.Normal("mu_FeH", mu=-0.20, sigma=0.08),
                                       pm.Normal("mu_MgFe", mu=0.42, sigma=0.02)
                                       ])
        mu = pm.Deterministic("shifts", mu_components, dims="features")

        noise_stars_in_single_prog = pm.MvNormal("noise", 
                                                  mu, 
                                                  chol=chol, 
                                                  dims=("star_id", "features"))
        

        return model

def sample_noise_training(model, 
                          n_samples,
                          random_seed):


    with model:
        prior_samples = pm.sample_prior_predictive(samples=n_samples,
                                                   random_seed=random_seed)
        noise_matrix = prior_samples.prior["noise"].values[0]
        
    return noise_matrix


def oversample_data(X,Y):
    
    x_idx = np.arange(len(X))
    class_x = np.zeros_like(x_idx)
    bin_edges = [[6,8], [8,9], [9,11]] # stellar mass   
    masks = [((Y[:,1]>be[0])&(Y[:,1]<be[1])) for be in bin_edges]

    for i,mask in enumerate(masks):
        class_x[mask] = i
        
    for c in set(class_x):
        f = len(class_x[class_x==c]) / len(class_x)
        print(f"Class {c}: {f:.2f}")

    print("="*30)
    ros = RandomOverSampler(random_state=42)
    
    x_idx_new, new_class_x = ros.fit_resample(x_idx[:,None], class_x)
    #
    for c in set(new_class_x):
        f = len(new_class_x[new_class_x==c]) / len(new_class_x)
        print(f"Class {c}: {f:.2f}")
        
    return X[x_idx_new][:,0], Y[x_idx_new][:,0]

def shuffle_axis1_independently(array):
    """
    Shuffles the order of elements along axis 1 (the 100 dimension)
    independently for each slice along axis 0 (the N dimension).
    """
    N, D2, D3 = array.shape
    
    # Create an array of random permutations for the 100 positions.
    # The shape will be (N, D2).
    # Each row will contain a unique shuffle of [0, 1, ..., 99].
    # We use np.arange(D2) to get the base indices [0, 1, ..., 99].
    permutations = np.stack([
        np.random.permutation(D2) for _ in range(N)
    ])
    
    # Now, we use advanced indexing to apply these permutations.
    # 1. np.arange(N) creates the row indices [0, 1, ..., N-1].
    # 2. We expand it to (N, 1) and broadcast it to (N, D2).
    # 3. This gives us the index pairs (i, permutations[i, j]) for the new array.
    
    # We need a meshgrid to match the row index (Ni) to its corresponding permutation.
    # row_indices: [[0, 0, ..., 0], [1, 1, ..., 1], ...] (Shape N x 100)
    row_indices = np.arange(N)[:, np.newaxis]
    
    # Apply the advanced indexing:
    # array[row_indices, permutations, :]
    shuffled_array = array[row_indices, permutations, :]
    
    return shuffled_array