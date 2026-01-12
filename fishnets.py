import flax.linen as nn
import jax
import jax.numpy as jnp
import math
import numpy as np
import optax
import pickle
import pymc as pm
import tensorflow_probability.substrates.jax as tfp
from tqdm import tqdm
from typing import Sequence, Any, Callable
from sklearn.preprocessing import RobustScaler
Array = Any


# Print JAX device
print(jax.devices(), flush=True)

import os
os.environ['TF_GPU_ALLOCATOR'] = "cuda_malloc_async"

key = jax.random.PRNGKey(0) # pseudo-random key for Jax network.
key, dropout_key = jax.random.split(key=key, num=2)

def fill_triangular(x):
    m = x.shape[0] # should be n * (n+1) / 2
    # solve for n
    n = int(math.sqrt((0.25 + 2 * m)) - 0.5)
    idx = jnp.array(m - (n**2 - m))
    x_tail = x[idx:]

    return jnp.concatenate([x_tail, jnp.flip(x, [0])], 0).reshape(n, n)


def fill_diagonal(a, val):
  assert a.ndim >= 2
  i, j = jnp.diag_indices(min(a.shape[-2:]))
  return a.at[..., i, j].set(val)


def construct_fisher_matrix_single(outputs):
    Q = tfp.math.fill_triangular(outputs)
    middle = jnp.diag(jnp.triu(Q) - nn.softplus(jnp.triu(Q)))
    padding = jnp.zeros(Q.shape)
    L = Q - fill_diagonal(padding, middle)

    return jnp.einsum('...ij,...jk->...ik', L, jnp.transpose(L, (1, 0)))


class MLP(nn.Module):
  features: Sequence[int]
  act: Callable = nn.elu

  @nn.compact
  def __call__(self, x, training: bool = True):
    for feat in self.features[:-1]:
            x = self.act(nn.Dense(feat)(x))
            x = nn.Dropout(rate=0.25, deterministic=not training)(x)
    x = nn.Dense(self.features[-1])(x)
    return x



class FishnetNetwork(nn.Module):
    hidden_channels: list
    n_p: int
    act: Callable = nn.leaky_relu
    input_dim: int = 8
    

    def setup(self):

        fdim = self.n_p + ((self.n_p * (self.n_p + 1)) // 2)
        fisherdim = ((self.n_p * (self.n_p + 1)) // 2)
        self.scorenet = MLP(self.hidden_channels + (self.n_p,), 
                            act=self.act)
        self.fishernet = MLP(self.hidden_channels + (fisherdim,), 
                             act=self.act)

    def __call__(self, x, training=True):

        # get score embedding and then sum
        # mask out dummy indices before sum
        score = self.scorenet(x, training=training).sum(0)
        # get fisher cholesky factors
        fisher = self.fishernet(x, training=training)

        # get the fisher matrix and add prior term
        fisher = jax.vmap(construct_fisher_matrix_single)(fisher)
        # mask out dummy indices before sum
        fisher = jnp.sum(fisher, axis=0) + jnp.eye(self.n_p)
        
        fisher = fisher + 1e-6 * jnp.eye(self.n_p)
        
        # get MLE
        x = jnp.linalg.solve(fisher, score)
        
        return x, score, fisher
        
    """def __call__(self, x,  training=True):

        # get score embedding and then sum
        score = self.scorenet(x,training=training).sum(0)

        # get fisher cholesky factors
        fisher = self.fishernet(x,training=training)

        # get the fisher matrix and add prior term
        fisher = jax.vmap(construct_fisher_matrix_single)(fisher)

        fisher = jnp.sum(fisher, axis=0) + jnp.eye(self.n_p)

        # get MLE
        x = jnp.einsum('...jk,...k->...j', jnp.linalg.inv(fisher), score)

        return x, score, fisher"""




        
class FISHNET():
   
    def __init__(self, 
                 n_params: int, 
                 n_d: int,
                 n_features: int,
                 n_hidden_layers: int = 3,
                 n_nodes_per_layer: int = 128):
        
        # Initialise parameters
        self.n_params = n_params # number of parameters to infer
        self.n_d = n_d # number of data points
        self.n_features = n_features # dimensionality of each data point
      
        # Initialise model
        self.model = FishnetNetwork(hidden_channels=[n_nodes_per_layer] * n_hidden_layers,
                                    n_p=n_params)
        # Initialise weights
        init_rngs = {'params': key, 'dropout': key}
        self.w = self.model.init(init_rngs, jnp.ones((n_d,n_features)), training=True)
        
    def __call__(self, 
                 data: jnp.ndarray):
        
        # Apply function
        _app = lambda d: self.model.apply(self.w, d, rngs={'dropout': dropout_key}, training=False)
        mle_pred, score, fisher = jax.vmap(_app)(data.reshape(-1, self.n_d, self.n_features)[:])

        return np.array(mle_pred), score, fisher

    def train(self, 
              data_: np.ndarray,
              theta_: np.ndarray,
              val_data_: np.ndarray,
              val_theta_: np.ndarray,
              noise_list: np.ndarray,
              obs_noise_list: np.array,
              data_scaler,
              lr: float = 1e-4,
              batch_size: int = 200,
              epochs: int = 3000,
              burn_in: int = 200,
              weights_dir: str = None
              ):
        
        rng = np.random.default_rng(42)
        
        n_cal_noise = noise_list.shape[0]
        n_obs_noise = obs_noise_list.shape[0]
        n_data = data_.shape[0]
        n_val_data = val_data_.shape[0]
        
        # Create a different noise configuration each epoch
        key_noise = jax.random.PRNGKey(998)
        cal_noise_idx = rng.integers(0, n_cal_noise, size=(epochs,n_data))
        obs_noise_idx = rng.integers(0, n_obs_noise, size=(epochs,n_data*100))

        # Repeat for validation data
        cal_noise_idx_val = rng.integers(0, n_cal_noise, size=n_val_data)
        obs_noise_idx_val = rng.integers(0, n_obs_noise, size=n_val_data*100)

        
        # Scale train and validation parameters
        theta_scaler = RobustScaler()
        
        theta_ = theta_scaler.fit_transform(theta_)
        val_theta_ = theta_scaler.transform(val_theta_)
        
        # Add calibration deviations
        val_data_ = val_data_ + noise_list[cal_noise_idx_val]
        # Add observation noise
        obs_err = obs_noise_list[obs_noise_idx_val].reshape(n_val_data,100,4)
        val_data_ = np.concatenate([val_data_, obs_err], axis=2)

        # Scale validation data
        val_data_ = data_scaler.transform(val_data_.reshape(-1,8)).reshape(-1,100,8)

        # Initialise loss function  
        def kl_loss(w, x_batched, theta_batched, rng_key):
            def fn(x, theta, k):
                mle, score, F = self.model.apply(w, x, training=True, rngs={'dropout': k})
                return mle, F
            
            batch_keys = jax.random.split(rng_key, x_batched.shape[0])
            mle, F = jax.vmap(fn)(x_batched, theta_batched, batch_keys)
            # Calculate the quadratic term: (theta - mle)^T * F * (theta - mle)
            diff = theta_batched - mle
            # Use einsum for cleaner batch multiplication
            quad_term = jnp.einsum('ij,ijk,ik->i', diff, F, diff)

            # Calculate Log Determinant safely using signs and logs (slogdet)
            # jnp.linalg.slogdet returns (sign, logabsdet). 
            # Since F is a Fisher matrix, it must be Positive Definite, so sign should be 1.
            sign, logdet = jnp.linalg.slogdet(F)

            # If the optimization goes wild, F might not be PD strictly numerically. 
            # slogdet is safer than log(det).
            
            # Add penalty term to prevent collapsing to mean
            theta_mean = jnp.mean(theta_batched, axis=0)
            # MSE of MLEs from the mean per theta dimension excluding noise
            mle_mean_squared_diff = jnp.mean( (mle - theta_mean)**2, axis=0)
            # Minus log MSE penalty
            minus_log_MSE_per_dimension = - jnp.log(mle_mean_squared_diff + 1e-10)
            collapse_penalty = 0.1*jnp.sum(minus_log_MSE_per_dimension)
            
            # Combine
            loss = -jnp.mean(-0.5 * quad_term + 0.5 * logdet) + collapse_penalty
            
            # Optional: Add a check to return 0.0 or a dummy value if NaN 
            # (prevents one bad batch from killing the whole run, though clipping is better)
            return jax.lax.select(jnp.isnan(loss), 0.0, loss)
        
        # Initialise optimiser
        tx = optax.chain(optax.clip_by_global_norm(1.0),  # Clip gradients with norm > 1.0
                         optax.adam(learning_rate=lr)
        )
        opt_state = tx.init(self.w)
        loss_grad_fn = jax.value_and_grad(kl_loss)

        # this is a hack to make the for-loop training much faster in jax
        def body_fun(i, inputs):
            w,loss_val, opt_state, _data, _theta, rng_key = inputs
            x_samples = _data[i]
            y_samples = _theta[i]
            
            # Split key: one for this step (step_key), one for the next loop (new_key)
            rng_key, step_key = jax.random.split(rng_key)

            loss_val, grads = loss_grad_fn(w, x_samples, y_samples, step_key)
            updates, opt_state = tx.update(grads, opt_state)
            w = optax.apply_updates(w, updates)

            return w, loss_val, opt_state, _data, _theta, rng_key
        
        # Initialise training loop
        key = jax.random.PRNGKey(999)
        loop_key = jax.random.PRNGKey(42)

        losses = jnp.zeros(epochs)
        loss_val = 0.
        val_losses = jnp.zeros(epochs)
        lower = 0
        upper = data_.shape[0] // batch_size

        pbar = tqdm(range(epochs), leave=True, position=0)

        print("Start training", flush=True)
        for j in pbar:
            
            # Add calibration deviations
            _data = data_ + noise_list[cal_noise_idx[j]]

            # Add observation noise
            if j>=burn_in:
                obs_err = obs_noise_list[obs_noise_idx[j]].reshape(_data.shape[0],100,4)
                _data = np.concatenate([_data, obs_err], axis=2)
            else:
                _data = np.concatenate([data_, np.zeros_like(data_)],axis=2)

            # Scale data
            _data = data_scaler.transform(_data.reshape(-1,8)) 
            
            # Reshape data into the progenitor groups
            _data = _data.reshape(-1,100,8)

            _data = jnp.array(_data)

            # Select a number of training examples which is divisible by the batch size
            n_train = (_data.shape[0]//batch_size)*batch_size

            # shuffle data every epoch
            key,rng = jax.random.split(key)
            randidx = jax.random.permutation(key, jnp.arange(theta_.reshape(-1, self.n_params).shape[0]), independent=True)[:n_train]
            
            _data = _data.reshape(-1, self.n_d, 8)[randidx].reshape( -1, batch_size, self.n_d, 8)
            _theta = theta_.reshape(-1, self.n_params)[randidx].reshape(-1, batch_size, self.n_params)            

            inits = (self.w, loss_val, opt_state, _data, _theta, loop_key)

            # Perform training for all batches
            self.w, loss_val, opt_state, _data, _theta, loop_key = jax.lax.fori_loop(lower,upper,body_fun,inits)

            if weights_dir is not None:
                # Save weights
                pickle.dump(self.w, open(f"{weights_dir}/epoch_{j}.pkl","wb"))

            losses = losses.at[j].set(loss_val)

            if val_data_ is not None and val_theta_ is not None:
                
                """_app = lambda d: self.model.apply(self.w, d, rngs={'dropout': dropout_key}, training=False)
                mle, score, F = jax.vmap(_app)(val_data_.reshape(-1, self.n_d, self.n_features)[:])

                val_loss = -jnp.mean(-0.5 * jnp.einsum('ij,ij->i', (val_theta_ - mle), \
                                                        jnp.einsum('ijk,ik->ij', F, (val_theta_ - mle))) \
                                                        + 0.5*jnp.log(jnp.linalg.det(F)), axis=0)"""
                
                @jax.jit
                def kl_loss_val(w, x_batched, theta_batched, rng_key):

                    def fn(x, theta, k):
                        mle,score,F = self.model.apply(w, x, rngs={'dropout': k}, training=False)
                        return mle, F
                    
                    # Split keys for vmap so every sample gets a different dropout mask
                    batch_keys = jax.random.split(rng_key, x_batched.shape[0])
                    
                    mle, F = jax.vmap(fn)(x_batched, theta_batched, batch_keys)

                    return -jnp.mean(-0.5 * jnp.einsum('ij,ij->i', (theta_batched - mle), \
                                                            jnp.einsum('ijk,ik->ij', F, (theta_batched - mle))) \
                                                                + 0.5*jnp.log(jnp.linalg.det(F)), axis=0)
                dummy_key = jax.random.PRNGKey(0)
                val_loss = kl_loss_val(self.w, val_data_, val_theta_, dummy_key)
                  
                val_losses = val_losses.at[j].set(val_loss)
                pbar.set_description('epoch %d loss: %.5f val_loss: %.5f'%(j, loss_val, val_loss))
                
            else:
                pbar.set_description('epoch %d loss: %.5f'%(j, loss_val))

        training_results = {"epochs": epochs,
                            "losses": losses,
                            "val_losses": val_losses}

        return training_results


