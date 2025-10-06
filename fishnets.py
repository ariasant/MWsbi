import flax.linen as nn
import jax
import jax.numpy as jnp
import math
import matplotlib.pyplot as plt
import numpy as np
import optax
import pickle
import pymc as pm
import tensorflow_probability.substrates.jax as tfp
from tqdm import tqdm
from typing import Sequence, Any, Callable
Array = Any


# Print JAX device
print(jax.devices(), flush=True)

import os
os.environ['TF_GPU_ALLOCATOR'] = "cuda_malloc_async"



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
  def __call__(self, x):
    for feat in self.features[:-1]:
      x = self.act(nn.Dense(feat)(x))
    x = nn.Dense(self.features[-1])(x)
    return x



class FishnetNetwork(nn.Module):
    hidden_channels: list
    n_p: int
    act: Callable = nn.leaky_relu

    def setup(self):

        fdim = self.n_p + ((self.n_p * (self.n_p + 1)) // 2)
        fisherdim = ((self.n_p * (self.n_p + 1)) // 2)
        self.scorenet = MLP(self.hidden_channels + (self.n_p,), act=self.act)
        self.fishernet = MLP(self.hidden_channels + (fisherdim,), act=self.act)

    def __call__(self, x, mask=None):

        # get score embedding and then sum
        # mask out dummy indices before sum
        score = self.scorenet(x).sum(0)
        # get fisher cholesky factors
        fisher = self.fishernet(x)

        # get the fisher matrix and add prior term
        fisher = jax.vmap(construct_fisher_matrix_single)(fisher)
        # mask out dummy indices before sum
        fisher = jnp.sum(fisher, axis=0) + jnp.eye(self.n_p)

        # mask out

        # get MLE
        x = jnp.einsum('...jk,...k->...j', jnp.linalg.inv(fisher), score)

        return x, score, fisher


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
        key = jax.random.PRNGKey(0) # pseudo-random key for Jax network.
        
        self.model = FishnetNetwork(hidden_channels=[n_nodes_per_layer] * n_hidden_layers,
                                    n_p=n_params)
        # Initialise weights
        self.w = self.model.init(key, jnp.ones((n_d,n_params)))
        
    def __call__(self, 
                 data: jnp.ndarray):
        
        # Apply function
        _app = lambda d: self.model.apply(self.w,d)  

        mle_pred, score, fisher = jax.vmap(_app)(data.reshape(-1, self.n_d, self.n_params)[:])

        return np.array(mle_pred), score, fisher

    def train(self, 
              data_: np.ndarray,
              theta_: np.ndarray,
              val_data_: np.ndarray = None,
              val_theta_: np.ndarray = None,
              noise_list: np.ndarray = None,
              obs_noise_list: np.array = None,
              data_scaler = None,
              lr: float = 1e-4,
              batch_size: int = 200,
              epochs: int = 3000,
              weights_dir: str = None
              ):
        
        # Convert data to JAX arrays
        data_ = data_
        theta_ = theta_
        noise_list = noise_list
        obs_noise_list = obs_noise_list


        if val_data_ is not None:
            val_data_ = jnp.array(val_data_)
            val_theta_ = jnp.array(val_theta_)

            # Scale data
            val_data_ = data_scaler.transform(val_data_.reshape(-1,self.n_features)).reshape(-1,100,self.n_features)
            # Shuffle
            key = jax.random.PRNGKey(9900)
            # Select a number of training examples which is divisible by the batch size
            n_val = (val_data_.shape[0]//batch_size)*batch_size
            # shuffle data 
            val_randidx = jax.random.permutation(key, jnp.arange(val_theta_.reshape(-1, self.n_params).shape[0]), independent=True)[:n_val]
           


        # Initialise loss function
        @jax.jit
        def kl_loss(w, x_batched, theta_batched):

            def fn(x, theta):
                mle,score,F = self.model.apply(w, x)
                return mle, F

            mle, F = jax.vmap(fn)(x_batched, theta_batched)

            return -jnp.mean(-0.5 * jnp.einsum('ij,ij->i', (theta_batched - mle), \
                                                    jnp.einsum('ijk,ik->ij', F, (theta_batched - mle))) \
                                                        + 0.5*jnp.log(jnp.linalg.det(F)), axis=0)
        # Initialise optimiser
        tx = optax.adam(learning_rate=lr)
        opt_state = tx.init(self.w)
        loss_grad_fn = jax.value_and_grad(kl_loss)

        # this is a hack to make the for-loop training much faster in jax
        def body_fun(i, inputs):
            w,loss_val, opt_state, _data, _theta = inputs
            x_samples = _data[i]
            y_samples = _theta[i]

            loss_val, grads = loss_grad_fn(w, x_samples, y_samples)
            updates, opt_state = tx.update(grads, opt_state)
            w = optax.apply_updates(w, updates)

            return w, loss_val, opt_state, _data, _theta
        
        # Initialise training loop
        key = jax.random.PRNGKey(999)

        losses = jnp.zeros(epochs)
        loss_val = 0.
        val_losses = jnp.zeros(epochs)
        lower = 0
        upper = data_.shape[0] // batch_size

        pbar = tqdm(range(epochs), leave=True, position=0)

        print("Start training", flush=True)
        n_cal_noise = noise_list.shape[0]
        n_obs_noise = obs_noise_list.shape[0]
        n_data = data_.shape[0]

        # Create a different noise configuration each epoch
        key_noise = jax.random.PRNGKey(998)
        # See which progenitors to consider with the alternative potenital model
        cal_noise_idx = jax.random.randint(key_noise, 
                                           shape=(epochs,n_data), 
                                           minval=0, maxval=n_cal_noise)
        obs_noise_idx = jax.random.randint(key_noise+12, 
                                           shape=(epochs,n_data*100), 
                                           minval=0, maxval=n_obs_noise)

        for j in pbar:
            
            # Add calibration deviations
            _data = data_ + noise_list[cal_noise_idx[j]]
            # Add observation noise
            _data = _data.reshape(-1,self.n_features) + obs_noise_list[obs_noise_idx[j]]

            # Scale data
            _data = data_scaler.transform(_data) 
            
            # Reshape data into the progenitor groups
            _data = _data.reshape(-1,100,self.n_features)

            _data = jnp.array(_data)

            # Select a number of training examples which is divisible by the batch size
            n_train = (_data.shape[0]//batch_size)*batch_size

            # shuffle data every epoch
            key,rng = jax.random.split(key)
            randidx = jax.random.permutation(key, jnp.arange(theta_.reshape(-1, self.n_params).shape[0]), independent=True)[:n_train]
            
            _data = _data.reshape(-1, self.n_d, self.n_features)[randidx].reshape(batch_size, -1, self.n_d, self.n_features)
            _theta = theta_.reshape(-1, self.n_params)[randidx].reshape(batch_size, -1, self.n_params)            

            inits = (self.w, loss_val, opt_state, _data, _theta)

            self.w, loss_val, opt_state, _data, _theta = jax.lax.fori_loop(lower, upper, body_fun, inits)

            if weights_dir is not None:
                # Save weights
                pickle.dump(self.w, open(f"{weights_dir}/epoch_{j}.pkl","wb"))

            losses = losses.at[j].set(loss_val)

            if val_data_ is not None and val_theta_ is not None:

                _val_data = val_data_.reshape(-1, self.n_d, self.n_features)[val_randidx].reshape(batch_size, -1, self.n_d, self.n_features)
                _val_theta = val_theta_.reshape(-1, self.n_params)[val_randidx].reshape(batch_size, -1, self.n_params)
                
                _val_data = _val_data.reshape(-1, _val_data.shape[2], _val_data.shape[3])
                _val_theta = _val_theta.reshape(-1, _val_theta.shape[-1])

                # calculate validation loss
                val_loss = kl_loss(self.w, _val_data, _val_theta)
                val_losses = val_losses.at[j].set(val_loss)
                pbar.set_description('epoch %d loss: %.5f val_loss: %.5f'%(j, loss_val, val_loss))
            else:
                pbar.set_description('epoch %d loss: %.5f'%(j, loss_val))

        training_results = {"epochs": epochs,
                            "losses": losses,
                            "val_losses": val_losses}

        return training_results


      


       

       
