
import flax.linen as nn
import jax
import jax.numpy as jnp
import math
import matplotlib.pyplot as plt
import numpy as np
import optax
import tensorflow_probability.substrates.jax as tfp
from tqdm import tqdm
from typing import Sequence, Any, Callable
Array = Any



def compute_mmd(x, y, sigma=1.0):
    xx = jnp.dot(x, x.T)
    yy = jnp.dot(y, y.T)
    xy = jnp.dot(x, y.T)

    X_sq = jnp.expand_dims(jnp.diag(xx), 1) + jnp.expand_dims(jnp.diag(xx), 0) - 2 * xx
    Y_sq = jnp.expand_dims(jnp.diag(yy), 1) + jnp.expand_dims(jnp.diag(yy), 0) - 2 * yy
    XY_sq = jnp.expand_dims(jnp.sum(x**2, axis=1), 1) + jnp.expand_dims(jnp.sum(y**2, axis=1), 0) - 2 * xy

    X_exp = jnp.exp(-X_sq / (2 * sigma**2))
    Y_exp = jnp.exp(-Y_sq / (2 * sigma**2))
    XY_exp = jnp.exp(-XY_sq / (2 * sigma**2))

    loss = (jnp.mean(X_exp) + jnp.mean(Y_exp) - 2 * jnp.mean(XY_exp)) * 0.5
    return loss

def fishnet_loss(theta_fid, mle, F):
    
    return -jnp.mean(
                -0.5 * jnp.einsum('ij,ij->i', (theta_fid - mle),
                                jnp.einsum('ijk,ik->ij', F, (theta_fid - mle)))
                + 0.5 * jnp.log(jnp.linalg.det(F)), axis=0
            )


# Print JAX device
print(jax.devices(), flush=True)

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


"""class MLP(nn.Module):
  features: Sequence[int]
  act: Callable = nn.elu

  @nn.compact
  def __call__(self, x):
    for feat in self.features[:-1]:
      x = self.act(nn.Dense(feat)(x))
    x = nn.Dense(self.features[-1])(x)
    return x"""

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
            # Use [256, 128, 64, 32] as hidden layers, keeping input/output as before
            score_layers = [512, 256, 128, 64, 32, self.n_p]
            fisher_layers = [512, 256, 128, 64, 32, fisherdim]
            self.scorenet = MLP(score_layers, act=self.act)
            self.fishernet = MLP(fisher_layers, act=self.act)

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
              data_sim: np.ndarray,
              theta_sim: np.ndarray,
              data_obs: np.ndarray,
              val_data_sim: np.ndarray = None,
              val_theta_sim: np.ndarray = None,
              val_data_obs: np.ndarray = None,
              lr: float = 1e-4,
              batch_size: int = 200,
              epochs: int = 3000
              ):
        
        # Convert data to JAX arrays
        data_sim = jnp.array(data_sim)
        theta_sim = jnp.array(theta_sim)
        data_obs = jnp.array(data_obs)


        if val_data_sim is not None:
            val_data_sim = jnp.array(val_data_sim)
            val_data_sim.reshape(-1, val_data_sim.shape[0], self.n_d, self.n_features)
            val_theta_sim = jnp.array(val_theta_sim)
            val_theta_sim.reshape(-1, val_data_sim.shape[0], self.n_params)
            val_data_obs = jnp.array(val_data_obs)
            val_data_obs.reshape(-1, val_data_obs.shape[0], self.n_d, self.n_features)



        @jax.jit
        def kl_loss(w, x_sim, theta_sim, x_obs, mmd_lambda=0.):

            # Standard Fishnet loss (only on simulation samples)
            def fn(x):
                mle, score, F = self.model.apply(w, x)
                return mle, F

            mle_sim, F = jax.vmap(fn)(x_sim)
            fishnet_loss = -jnp.mean(
                -0.5 * jnp.einsum('ij,ij->i', (theta_sim - mle_sim),
                                jnp.einsum('ijk,ik->ij', F, (theta_sim - mle_sim)))
                + 0.5 * jnp.log(jnp.linalg.det(F)), axis=0
            )

            # MMD between simulation and observation data
            mle_obs, F = jax.vmap(fn)(x_obs)

            # Ignore nan values in the two arrays
            mmd_loss = compute_mmd(mle_sim, mle_obs)

            return fishnet_loss + mmd_lambda * mmd_loss
        

        # Initialise optimiser
        tx = optax.chain(
                optax.clip_by_global_norm(1.0),  # Clips gradients with a max global norm of 1.0
                optax.adam(learning_rate=lr)  # AdamW optimizer with weight decay
        )
        opt_state = tx.init(self.w)
        loss_grad_fn = jax.value_and_grad(kl_loss)

        # this is a hack to make the for-loop training much faster in jax
        def body_fun(i, inputs):
            w,loss_val, opt_state, _data_sim, _theta_sim, _data_obs = inputs
            x_samples_sim = _data_sim[i]
            y_samples_sim = _theta_sim[i]
            x_samples_obs = _data_obs[i]

            loss_val, grads = loss_grad_fn(w, x_samples_sim, y_samples_sim, x_samples_obs)
            updates, opt_state = tx.update(grads, opt_state)
            w = optax.apply_updates(w, updates)

            return w, loss_val, opt_state, _data_sim, _theta_sim, _data_obs
        
        # Initialise training loop
        key = jax.random.PRNGKey(999)

        losses = jnp.zeros(epochs)
        loss_val = 0.
        val_losses = jnp.zeros(epochs)
        lower = 0
        upper = data_sim.shape[0] // batch_size

        pbar = tqdm(range(epochs), leave=True, position=0)

        train_mle_obs = jax.vmap(lambda d: self.model.apply(self.w, d)[0])(data_obs)
        train_mle_sim = jax.vmap(lambda d: self.model.apply(self.w, d)[0])(data_sim)

        train_mmd = compute_mmd(train_mle_sim, train_mle_obs)
        print(f"Starting MMD: {train_mmd:.2f}", flush=True)

        sim_outputs = jax.vmap(lambda d: self.model.apply(self.w, d))(val_data_sim)
        val_mle_sim = sim_outputs[0]
        val_mle_obs = jax.vmap(lambda d: self.model.apply(self.w, d)[0])(val_data_obs)
        mmd_start = compute_mmd(val_mle_sim, val_mle_obs)
        print(f"Starting val MMD: {mmd_start:.2f}", flush=True)

        for j in pbar:
            
            key,rng = jax.random.split(key)
       
            randidx = jax.random.permutation(key, jnp.arange(theta_sim.reshape(-1, self.n_params).shape[0]), independent=True)
            # Shuffle the data
            data_sim = data_sim[randidx]
            theta_sim = theta_sim[randidx]
            data_obs = data_obs[randidx]

            # Define number of batches, the same number of batches are taken from the observations and simulations
            n_batches = len(data_sim) // batch_size
            # Split data into batches
            _data_sim = data_sim[:batch_size * n_batches].reshape(-1, batch_size, self.n_d, self.n_features)
            _theta_sim = theta_sim[:batch_size * n_batches].reshape(-1, batch_size, self.n_params)
            _data_obs = data_obs[:batch_size * n_batches].reshape(-1, batch_size, self.n_d, self.n_features)

            inits = (self.w, loss_val, opt_state, _data_sim, _theta_sim, _data_obs)

            self.w, loss_val, opt_state, _data_sim, _theta_sim, _data_obs = jax.lax.fori_loop(lower, upper, body_fun, inits)

            losses = losses.at[j].set(loss_val)


            train_mle_obs = jax.vmap(lambda d: self.model.apply(self.w, d)[0])(data_obs)
            train_mle_sim = jax.vmap(lambda d: self.model.apply(self.w, d)[0])(data_sim)

            train_mmd = compute_mmd(train_mle_sim, train_mle_obs)


            if val_data_sim is not None and val_theta_sim is not None:                          

                sim_outputs = jax.vmap(lambda d: self.model.apply(self.w, d))(val_data_sim)
                val_mle_sim = sim_outputs[0]
                F_sim = sim_outputs[2]
                val_mle_obs = jax.vmap(lambda d: self.model.apply(self.w, d)[0])(val_data_obs)

                # Ignore nan values in the two distributions
                val_mle_obs = val_mle_obs[~jnp.isnan(val_mle_obs).any(axis=-1)]
                val_mle_sim = val_mle_sim[~jnp.isnan(val_mle_sim).any(axis=-1)]

                n_samples = min(len(val_mle_sim), len(val_mle_obs))

                val_mmd = compute_mmd(val_mle_sim[:n_samples], val_mle_obs[:n_samples])
                val_loss = fishnet_loss(val_theta_sim, val_mle_sim, F_sim)

                val_losses = val_losses.at[j].set(val_loss)      

                pbar.set_description('epoch %d loss: %.5f mmd: %.5f val_loss: %.5f val_mmd: %.5f'%(j, loss_val, train_mmd, val_loss, val_mmd))

            else:
                pbar.set_description('epoch %d loss: %.5f mmd: %.5f'%(j, loss_val, train_mmd))

        training_results = {"epochs": epochs,
                            "losses": losses,
                            "val_losses": val_losses}

        return training_results


      


       

       