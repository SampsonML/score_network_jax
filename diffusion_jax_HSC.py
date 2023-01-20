#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
A jax implementation of the diffusion model from 
the paper "Improved Techniques for Training Score-Based Generative Models"
https://arxiv.org/abs/2006.09011
Code taken primarily from https://github.com/yang-song/score_sde/
Modifications by Matt Sampson include:
    - Minor updates to use the latest version of flax
    - changed optim.Adam to optax.adam (required for latex flax)
"""

# require newest cuda supported jaxlib for GPU/TPU use
#!conda install cudatoolkit-dev=11.3.1 -c conda-forge
#!pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

get_ipython().run_line_magic('matplotlib', 'inline')
import functools
import math
import string
from typing import Any, Sequence, Optional
import flax.linen as nn
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.nn.initializers as init
import matplotlib as mpl
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import flax
import optax
Path("outputs").mkdir(exist_ok=True)

# test we can find correct device
from jax.lib import xla_bridge
print(f'Device found is: {xla_bridge.get_backend().platform}')


# In[2]:


"""Common layers for defining score networks.
"""
class InstanceNorm2dPlus(nn.Module):
  """InstanceNorm++ as proposed in the original NCSN paper."""
  bias: bool = True

  @staticmethod
  def scale_init(key, shape, dtype=jnp.float32):
    normal_init = init.normal(0.02)
    return normal_init(key, shape, dtype=dtype) + 1.

  @nn.compact
  def __call__(self, x):
    means = jnp.mean(x, axis=(1, 2))
    m = jnp.mean(means, axis=-1, keepdims=True)
    v = jnp.var(means, axis=-1, keepdims=True)
    means_plus = (means - m) / jnp.sqrt(v + 1e-5)

    h = (x - means[:, None, None, :]) / jnp.sqrt(jnp.var(x, axis=(1, 2), keepdims=True) + 1e-5)

    h = h + means_plus[:, None, None, :] * self.param('alpha', InstanceNorm2dPlus.scale_init, (1, 1, 1, x.shape[-1]))
    h = h * self.param('gamma', InstanceNorm2dPlus.scale_init, (1, 1, 1, x.shape[-1]))
    if self.bias:
      h = h + self.param('beta', init.zeros, (1, 1, 1, x.shape[-1]))

    return h


def ncsn_conv1x1(x, out_planes, stride=1, bias=True, dilation=1, init_scale=1.):
  """1x1 convolution with PyTorch initialization. Same as NCSNv1/v2."""
  init_scale = 1e-10 if init_scale == 0 else init_scale
  kernel_init = jnn.initializers.variance_scaling(1 / 3 * init_scale, 'fan_in',
                                                  'uniform')
  kernel_shape = (1, 1) + (x.shape[-1], out_planes)
  bias_init = lambda key, shape: kernel_init(key, kernel_shape)[0, 0, 0, :]
  output = nn.Conv(out_planes, kernel_size=(1, 1),
                   strides=(stride, stride), padding='SAME', use_bias=bias,
                   kernel_dilation=(dilation, dilation),
                   kernel_init=kernel_init,
                   bias_init=bias_init)(x)
  return output

def ncsn_conv3x3(x, out_planes, stride=1, bias=True, dilation=1, init_scale=1.):
  """3x3 convolution with PyTorch initialization. Same as NCSNv1/NCSNv2."""
  kernel_init = jnn.initializers.variance_scaling(1 / 3 * init_scale, 'fan_in',
                                                  'uniform')
  kernel_shape = (3, 3) + (x.shape[-1], out_planes)
  #bias_init = lambda key, shape: kernel_init(key, kernel_shape)[0, 0, 0, :]
  bias_init = jnn.initializers.zeros
  output = nn.Conv(out_planes,
                   kernel_size=(3, 3),
                   strides=(stride, stride),
                   padding='SAME',
                   use_bias= bias,
                   kernel_dilation=(dilation, dilation),
                   kernel_init=kernel_init,
                   bias_init=bias_init)(x)
  return output

# ---------------------------------------------------------------- #
# Functions below are ported over from the NCSNv1/NCSNv2 codebase: #
# https://github.com/ermongroup/ncsn                               #
# https://github.com/ermongroup/ncsnv2                             #
# ---------------------------------------------------------------- #

class CRPBlock(nn.Module):
  """CRPBlock for RefineNet. Used in NCSNv2."""
  features: int
  n_stages: int
  act: Any = nn.relu

  @nn.compact
  def __call__(self, x):
    x = self.act(x)
    path = x
    for _ in range(self.n_stages):
      path = nn.max_pool(
        path, window_shape=(5, 5), strides=(1, 1), padding='SAME')
      path = ncsn_conv3x3(path, self.features, stride=1, bias=False)
      x = path + x
    return x


class RCUBlock(nn.Module):
  """RCUBlock for RefineNet. Used in NCSNv2."""
  features: int
  n_blocks: int
  n_stages: int
  act: Any = nn.relu

  @nn.compact
  def __call__(self, x):
    for _ in range(self.n_blocks):
      residual = x
      for _ in range(self.n_stages):
        x = self.act(x)
        x = ncsn_conv3x3(x, self.features, stride=1, bias=False)
      x = x + residual

    return x

class MSFBlock(nn.Module):
  """MSFBlock for RefineNet. Used in NCSNv2."""
  shape: Sequence[int]
  features: int
  interpolation: str = 'bilinear'

  @nn.compact
  def __call__(self, xs):
    sums = jnp.zeros((xs[0].shape[0], *self.shape, self.features))
    for i in range(len(xs)):
      h = ncsn_conv3x3(xs[i], self.features, stride=1, bias=True)
      if self.interpolation == 'bilinear':
        h = jax.image.resize(h, (h.shape[0], *self.shape, h.shape[-1]), 'bilinear')
      elif self.interpolation == 'nearest_neighbor':
        h = jax.image.resize(h, (h.shape[0], *self.shape, h.shape[-1]), 'nearest')
      else:
        raise ValueError(f'Interpolation {self.interpolation} does not exist!')
      sums = sums + h
    return sums

class RefineBlock(nn.Module):
  """RefineBlock for building NCSNv2 RefineNet."""
  output_shape: Sequence[int]
  features: int
  act: Any = nn.relu
  interpolation: str = 'bilinear'
  start: bool = False
  end: bool = False

  @nn.compact
  def __call__(self, xs):
    rcu_block = functools.partial(RCUBlock, n_blocks=2, n_stages=2, act=self.act)
    rcu_block_output = functools.partial(RCUBlock,
                                        features=self.features,
                                        n_blocks=3 if self.end else 1,
                                        n_stages=2,
                                        act=self.act)
    hs = []
    for i in range(len(xs)):
      h = rcu_block(features=xs[i].shape[-1])(xs[i])
      hs.append(h)

    if not self.start:
      msf = functools.partial(MSFBlock, features=self.features, interpolation=self.interpolation)
      h = msf(shape=self.output_shape)(hs)
    else:
      h = hs[0]

    crp = functools.partial(CRPBlock, features=self.features, n_stages=2, act=self.act)
    h = crp()(h)
    h = rcu_block_output()(h)
    return h

class ConvMeanPool(nn.Module):
  """ConvMeanPool for building the ResNet backbone."""
  output_dim: int
  kernel_size: int = 3
  biases: bool = True

  @nn.compact
  def __call__(self, inputs):
    output = nn.Conv(features=self.output_dim,
                    kernel_size=(self.kernel_size, self.kernel_size),
                    strides=(1, 1),
                    padding='SAME',
                    use_bias=self.biases)(inputs)
    output = sum([
      output[:, ::2, ::2, :], output[:, 1::2, ::2, :],
      output[:, ::2, 1::2, :], output[:, 1::2, 1::2, :]
    ]) / 4.
    return output


class MeanPoolConv(nn.Module):
  """MeanPoolConv for building the ResNet backbone."""
  output_dim: int
  kernel_size: int = 3
  biases: bool = True

  @nn.compact
  def __call__(self, inputs):
    output = inputs
    output = sum([
      output[:, ::2, ::2, :], output[:, 1::2, ::2, :],
      output[:, ::2, 1::2, :], output[:, 1::2, 1::2, :]
    ]) / 4.
    output = nn.Conv(
      features=self.output_dim,
      kernel_size=(self.kernel_size, self.kernel_size),
      strides=(1, 1),
      padding='SAME',
      use_bias=self.biases)(output)
    return output


class ResidualBlock(nn.Module):
  """The residual block for defining the ResNet backbone. Used in NCSNv2."""
  output_dim: int
  normalization: Any
  resample: Optional[str] = None
  act: Any = nn.elu
  dilation: int = 1

  @nn.compact
  def __call__(self, x):
    h = self.normalization()(x)
    h = self.act(h)
    if self.resample == 'down':
      h = ncsn_conv3x3(h, h.shape[-1], dilation=self.dilation)
      h = self.normalization()(h)
      h = self.act(h)
      if self.dilation > 1:
        h = ncsn_conv3x3(h, self.output_dim, dilation=self.dilation)
        shortcut = ncsn_conv3x3(x, self.output_dim, dilation=self.dilation)
      else:
        h = ConvMeanPool(output_dim=self.output_dim)(h)
        shortcut = ConvMeanPool(output_dim=self.output_dim, kernel_size=1)(x)
    elif self.resample is None:
      if self.dilation > 1:
        if self.output_dim == x.shape[-1]:
          shortcut = x
        else:
          shortcut = ncsn_conv3x3(x, self.output_dim, dilation=self.dilation)
        h = ncsn_conv3x3(h, self.output_dim, dilation=self.dilation)
        h = self.normalization()(h)
        h = self.act(h)
        h = ncsn_conv3x3(h, self.output_dim, dilation=self.dilation)
      else:
        if self.output_dim == x.shape[-1]:
          shortcut = x
        else:
          shortcut = ncsn_conv1x1(x, self.output_dim)
        h = ncsn_conv3x3(h, self.output_dim)
        h = self.normalization()(h)
        h = self.act(h)
        h = ncsn_conv3x3(h, self.output_dim)

    return h + shortcut

class ConditionalResidualBlock(nn.Module):
  """The noise-conditional residual block for building NCSNv1."""
  output_dim: int
  normalization: Any
  resample: Optional[str] = None
  act: Any = nn.elu
  dilation: int = 1

  @nn.compact
  def __call__(self, x, y):
    h = self.normalization()(x, y)
    h = self.act(h)
    if self.resample == 'down':
      h = ncsn_conv3x3(h, h.shape[-1], dilation=self.dilation)
      h = self.normalization(h, y)
      h = self.act(h)
      if self.dilation > 1:
        h = ncsn_conv3x3(h, self.output_dim, dilation=self.dilation)
        shortcut = ncsn_conv3x3(x, self.output_dim, dilation=self.dilation)
      else:
        h = ConvMeanPool(output_dim=self.output_dim)(h)
        shortcut = ConvMeanPool(output_dim=self.output_dim, kernel_size=1)(x)
    elif self.resample is None:
      if self.dilation > 1:
        if self.output_dim == x.shape[-1]:
          shortcut = x
        else:
          shortcut = ncsn_conv3x3(x, self.output_dim, dilation=self.dilation)
        h = ncsn_conv3x3(h, self.output_dim, dilation=self.dilation)
        h = self.normalization()(h, y)
        h = self.act(h)
        h = ncsn_conv3x3(h, self.output_dim, dilation=self.dilation)
      else:
        if self.output_dim == x.shape[-1]:
          shortcut = x
        else:
          shortcut = ncsn_conv1x1(x, self.output_dim)
        h = ncsn_conv3x3(h, self.output_dim)
        h = self.normalization()(h, y)
        h = self.act(h)
        h = ncsn_conv3x3(h, self.output_dim)

    return h + shortcut


# In[3]:


""" 
MATT:
Want to build a U-NET in jax which will have 5 layers (to match Song+2020)
Need to take into acount the noise scales - ie embedding the noise scale into the model
Build this with flax.linen (flax.linen as nn)?
"""
# grabbed from https://github.com/yang-song/score_sde/blob/main/models/ncsnv2.py

CondResidualBlock = ConditionalResidualBlock
conv3x3 = ncsn_conv3x3

class NCSNv2(nn.Module):
  """NCSNv2 model architecture"""
  #config: ml_collections.ConfigDict

  @nn.compact
  #def __call__(self, x, labels, train=True):
  def __call__(self, x, labels):
    
    # hard coding configs for now
    sigma_begin   = 1                     # noise scale max
    sigma_end     = 1e-2                  # noise scale min
    num_scales    = 10                     # number of noise scales
    sigmas        = jnp.exp(jnp.linspace(jnp.log(sigma_end), 
                              jnp.log(sigma_begin),num_scales))
    sigmas = jax.numpy.flip(sigmas)
    im_size       = 32                    # image size
    nf            = 128                   # number of filters
    act           = nn.elu                # activation function
    normalizer    = InstanceNorm2dPlus    # normalization function
    interpolation = 'bilinear'            # interpolation method for upsample
    
    # data already centered
    h = x
    
    # Begin the U-Net
    h = conv3x3(h, nf, stride=1, bias=True)

    # ResNet backbone
    h = ResidualBlock(nf, resample=None, act=act, normalization=normalizer)(h)
    layer1 = ResidualBlock(nf, resample=None, act=act, normalization=normalizer)(h)
    h = ResidualBlock(2 * nf, resample='down', act=act, normalization=normalizer)(layer1)
    layer2 = ResidualBlock(2 * nf, resample=None, act=act, normalization=normalizer)(h)
    h = ResidualBlock(2 * nf,
                      resample='down',
                      act=act,
                      normalization=normalizer,
                      dilation=2)(layer2)
    layer3 = ResidualBlock(2 * nf, resample=None, act=act, normalization=normalizer, dilation=2)(h)
    h = ResidualBlock(2 * nf,
                      resample='down',
                      act=act,
                      normalization=normalizer,
                      dilation=4)(layer3)
    layer4 = ResidualBlock(2 * nf, resample=None, act=act, normalization=normalizer, dilation=4)(h)
    # U-Net with RefineBlocks
    ref1 = RefineBlock(layer4.shape[1:3],
                       2 * nf,
                      act=act,
                      interpolation=interpolation,
                      start=True)([layer4])
    ref2 = RefineBlock(layer3.shape[1:3],
                       2 * nf,
                      interpolation=interpolation,
                      act=act)([layer3, ref1])
    ref3 = RefineBlock(layer2.shape[1:3],
                       2 * nf,
                      interpolation=interpolation,
                      act=act)([layer2, ref2])
    ref4 = RefineBlock(layer1.shape[1:3],
                      nf,
                      interpolation=interpolation,
                      act=act,
                      end=True)([layer1, ref3])

    h = normalizer()(ref4)
    h = act(h)
    h = conv3x3(h, x.shape[-1])

    # normlising the output
    used_sigmas = sigmas[labels].reshape(
        (x.shape[0], *([1] * len(x.shape[1:]))))
    
    return h / used_sigmas




# In[4]:


"""
The loss function for a noise dependent score model from Song+2020
"""
def anneal_dsm_score_estimation(params, model, samples, labels, sigmas, key):
    """
    Loss function for annealed score estimation
    -------------------------------------------
    Inputs: params - the model parameters
            model - the score neural network
            samples - the samples from the data distribution
            labels - the noise scale labels
            sigmas - the noise scales
            key - the jax random key

    Output: loss - the loss value
    """
    used_sigmas = sigmas[labels].reshape((samples.shape[0], 
                                          *([1] * len(samples.shape[1:]))))
    noise = jax.random.normal(key, samples.shape)
    perturbed_samples = samples + noise * used_sigmas
    target = -noise / used_sigmas**2
    scores = model.apply({'params': params}, perturbed_samples, labels)
    #losses = jnp.square(score - target)
    loss_1 = 1 / 2. * ((scores - target) ** 2).sum(axis=-1) #* used_sigmas.squeeze() ** anneal_power
    loss = loss_1 * used_sigmas**2 
    loss = jnp.mean(loss)
    return loss


# In[5]:


""" 
The training of the NCSNv2 model. Here define the training
parameters and initialise the model. Train on a small scale 
for testing before moving to the full scale on GPU HPC.
"""
# ----------- #
# model setup #
# ----------- #

# load in data  low res
"""
box_size = 31
dataname = 'sources_box' + str(box_size) + '.npy'     
dataset = np.load(dataname)

# perform zero-padding of the data to get desired dimensions
data_padded_31 = []
for i in range(len(dataset)):
    data_padded_tmp = np.pad(dataset[i], ((0,1),(0,1)), 'constant')
    data_padded_31.append(data_padded_tmp)
dataset = np.array( data_padded_31 )
"""

# load in data  high res
box_size = 61
dataname = 'sources_box' + str(box_size) + '.npy'     
dataset = np.load(dataname)

# perform zero-padding of the data to get desired dimensions
data_padded_61 = []
for i in range(len(dataset)):
    data_padded_tmp = np.pad(dataset[i], ((1,2),(1,2)), 'constant')
    data_padded_61.append(data_padded_tmp)
dataset = np.array( data_padded_61 )

# convert dataset to jax array
data_jax = jnp.array(dataset)
# expand dimensions for channel dim
data_jax = jax.numpy.expand_dims(data_jax, axis=-1)


# In[6]:


# ------------------------------ #
# visualisation for code testing #
# ------------------------------ #
import cmasher as cmr
score_map = cmr.iceburn
data_map = cmr.ember
def plot_evolve(params,sample,step, labels):
    gaussian_noise = jax.random.normal(key_seq, shape=sample.shape)
    scores = model.apply({'params' : params}, gaussian_noise, labels)
    scores2 = model.apply({'params' : params}, sample, labels)
    fig , ax = plt.subplots(2,2,figsize=(16, 12), facecolor='white',dpi = 70)
    plt.subplots_adjust(wspace=0.01)
    plt.subplot(2,2,1)
    plt.imshow(scores[0], cmap=score_map)
    #plt.colorbar()
    plt.title('Gaussian Noise',fontsize=36,pad=15)
    plt.ylabel('score', fontsize=40)
    plt.subplot(2,2,2)
    plt.imshow(scores2[0], cmap=score_map)
    cbar = plt.colorbar()
    cbar.set_label(r'$\nabla_x log \ p(\mathbf{\tilde{x}})$', rotation=270, fontsize = 20,labelpad= 25)
    plt.title('Galaxy',fontsize=36,pad=15)
    plt.subplot(2,2,3)
    plt.imshow(gaussian_noise[0], cmap=data_map)
    plt.ylabel('data', fontsize=40)
    plt.subplot(2,2,4)
    plt.imshow(sample[0], cmap=data_map)
    cbar = plt.colorbar()
    cbar.set_label(r'pixel density', rotation=270, fontsize = 20,labelpad= 25)
    super_name = 'training step ' + str(step) 
    plt.suptitle(super_name, fontsize = 40)
    plt.tight_layout()
    save_name = 'score_estimation_pre_training_step_' + str(step) + '.png'
    plt.savefig(save_name,facecolor='white',dpi=200)
    plt.close()


# In[7]:


# TODO: write a dataloader to load in mini-batches of data
# see https://wandb.ai/jax-series/simple-training-loop/reports/Writing-a-Training-Loop-in-JAX-FLAX--VmlldzoyMzA4ODEy

# model training and init params
key_seq     = jax.random.PRNGKey(42)                # random seed
n_epochs    = 50                                    # number of epochs
n_steps     = 30                                    # number of steps per epoch
batch_size  = 32                                    # batch size
lr          = 1e-4                                  # learning rate
im_size     = 64                                    # image size

# construct the training data
# TODO: convert this to proper dataloader
batch = jnp.array(range(0, batch_size))
training_data = data_jax[batch]

# define noise levels and noise params
sigma_begin = 1
sigma_end   = 0.01
num_scales  = 10
sigmas      = jnp.exp(jnp.linspace(jnp.log(sigma_end), 
                        jnp.log(sigma_begin),num_scales))
sigmas = jax.numpy.flip(sigmas)
labels = jax.random.randint(key_seq, (len(training_data),), 
                            minval=0, maxval=len(sigmas), dtype=jnp.int32)

# model init variables
#input_shape = (jax.local_device_count(), im_size, im_size, 1)  
#label_shape = input_shape[:1]
input_shape = training_data.shape
label_shape = labels.shape
fake_input  = jnp.zeros(input_shape)
fake_label  = jnp.zeros(label_shape, dtype=jnp.int32)
params_rng, dropout_rng = jax.random.split(key_seq)

# define and initialise model
model = NCSNv2()
variables = model.init({'params': params_rng}, fake_input, fake_label)
init_model_state, initial_params = variables.pop('params')

# define optimiser using latest flax standards
optimizer = optax.adam( learning_rate=lr, 
                        b1=0.9, 
                        b2=0.999, 
                        eps=1e-08, 
                        eps_root=0.0, 
                        mu_dtype=None ) 

# initialise model state
params = initial_params
model_state = optimizer.init(params)

# name loss function
loss_fn = anneal_dsm_score_estimation

# A simple update loop
train    = True
plot     = False
samples  = training_data
from tqdm import tqdm

if train:
  loss_vector = np.zeros(n_steps)
  for i in tqdm(range(n_steps), desc='training model'):
    grads = jax.grad(loss_fn)(params, model, samples, labels, sigmas, key_seq)
    updates, model_state = optimizer.update(grads, model_state)
    params = optax.apply_updates(params, updates)
    loss_vector[i] = loss_fn(params, model, samples, labels, sigmas, key_seq)
    #print(f'loss at step {i}: {loss_vector[i]}')
    # make plot to see evolution
    if (plot): plot_evolve(params, samples, i, labels)
  print(f'initial loss: {loss_vector[0]}')
  print(f'final loss: {loss_vector[-1]}')
  

if plot:
  fig , ax = plt.subplots(1,1,figsize=(12, 8), facecolor='white',dpi = 70)
  steps = range(0,n_steps)
  plt.plot(steps,loss_vector, alpha = 0.80, zorder=0)
  #plt.scatter(steps,loss_vector, s=20, zorder=1)
  plt.xlabel('training steps', fontsize = 30)
  plt.ylabel('cross-entropy loss (arb)', fontsize = 30)
  plt.tight_layout()
  plt.savefig('loss_evolution.png',facecolor='white',dpi=300)


# In[ ]:


# ------------------------- #
# langevin dynamic sampling #
# ------------------------- #

def anneal_Langevin_dynamics(x_mod, scorenet, params, sigmas, rng, n_steps_each=100, 
                                step_lr=0.000008,denoise=True):
    # initialise arrays for images and scores
    images = []
    scores  = []

    for c, sigma in enumerate(sigmas):
        labels = jax.numpy.ones(x_mod.shape[0],dtype=np.int8) * c
        step_size = step_lr * (sigma / sigmas[-1]) ** 2
        desc = 'sampling at noise level: ' + str(c + 1) + ' / ' + str(len(sigmas))
        for s in tqdm(range(n_steps_each),desc=desc):
            grad = scorenet.apply({'params' : params}, x_mod, labels)
            noise = jax.random.normal(rng, shape=x_mod.shape)
            x_mod = x_mod + step_size * grad + noise * np.sqrt(step_size * 2)
            # store the progress per step
            #if (c == len(sigmas) - 1):
            images.append(x_mod)
            scores.append(grad)

    if denoise:
        last_noise = (len(sigmas) - 1) * jax.numpy.ones(x_mod.shape[0], dtype=np.int8)
        last_grad = scorenet.apply({'params' : params}, x_mod, last_noise)
        x_mod = x_mod + sigmas[-1] ** 2 * last_grad
        images.append(x_mod)
        scores.append(last_grad)

    return images, scores


# In[ ]:


# ---------------- #
# testing sampling #
# ---------------- #
samples = data_jax[key_seq]
gaussian_noise = jax.random.normal(key_seq, shape=samples.shape)
images, scores = anneal_Langevin_dynamics(  gaussian_noise, 
                                            model, 
                                            params, 
                                            sigmas, 
                                            key_seq,
                                            n_steps_each=64, 
                                            denoise=True  )


# In[ ]:


images_array = np.array(images)
col_map = cmr.lilac
fig , ax = plt.subplots(2,5,figsize=(16, 7), facecolor='white',dpi = 70)
plt_idx = int( len(images_array) / 10 )
n_panels = 7
step_array = [0, 2, 4, 8, 16, 32, 64]
for i in range(n_panels):
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.subplot(1,n_panels,i + 1)
    step = step_array[i] * 10
    name = 'step ' + str(int( step / 10 ))
    plt.title(name, fontsize = 24)
    plt.imshow(images_array[step][0], cmap=col_map)
    plt.axis('off')
plt.tight_layout()
plt.savefig('langevin_evolution_panel7.png',facecolor='white',dpi=300)
plt.show()

