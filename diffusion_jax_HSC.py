#!/usr/bin/env python
# coding: utf-8

# In[6]:


"""
A jax implementation of the diffusion model from 
the paper "Improved Techniques for Training Score-Based Generative Models"
https://arxiv.org/abs/2006.09011
Code taken primarily from https://github.com/yang-song/score_sde/
Modifications by Matt Sampson include:
    - Minor updates to use the latest version of flax
"""

# temporary fix for flax.optim issue:
# https://stackoverflow.com/questions/73488909/attributeerror-module-flax-has-no-attribute-optim
#!pip uninstall flax -y
#!pip install flax==0.5.1
# proper fix here (https://flax.readthedocs.io/en/latest/advanced_topics/optax_update_guide.html)
# not yet implemented

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
Path("outputs").mkdir(exist_ok=True)


# In[7]:


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

""" Old version of ncsn_conv3x3 """
"""
def ncsn_conv3x3(x, out_planes, stride=1, bias=True, dilation=1, init_scale=1.):
  # 3x3 convolution with PyTorch initialization. Same as NCSNv1/NCSNv2.
  init_scale = 1e-10 if init_scale == 0 else init_scale
  kernel_init = jnn.initializers.variance_scaling(1 / 3 * init_scale, 'fan_in',
                                                  'uniform')
  kernel_shape = (3, 3) + (x.shape[-1], out_planes)
  bias_init = lambda key, shape: kernel_init(key, kernel_shape)[0, 0, 0, :]
  output = nn.Conv(out_planes,
                   kernel_size=(3, 3),
                   strides=(stride, stride),
                   padding='SAME',
                   use_bias=bias,
                   kernel_dilation=(dilation, dilation),
                   kernel_init=kernel_init,
                   bias_init=bias_init)(x)
  return output
"""

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

# How to call bias_init search this and see - flax outdated thing
# Make a github repo for this
# Get much more familiar with github
# add a requirement.txt to show which versions of modules are needed

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


# In[8]:


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
  def __call__(self, x, labels, train=True):
    
    # hard coding configs for now
    sigma_begin   = 1                     # noise scale max
    sigma_end     = 1e-2                  # noise scale min
    num_scales    = 10                     # number of noise scales
    sigmas        = jnp.exp(jnp.linspace(jnp.log(sigma_end), 
                              jnp.log(sigma_begin),num_scales))
    im_size       = 32                    # image size
    nf            = 128                   # number of filters
    act           = nn.elu                # activation function
    normalizer    = InstanceNorm2dPlus    # normalization function
    interpolation = 'bilinear'            # interpolation method for upsample
    
    # data already centered
    h = x
    
    # Begin the U-Net
    h = conv3x3(h, nf, stride=1, bias=True)
    print(f'size of h: {h.shape}')
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


# In[9]:


"""
The loss function for a noise dependent score model from Song+2020
"""
def anneal_dsm_score_estimation(model, samples, labels, sigmas, key, anneal_power=2.):
    sigmas = sigmas[..., None]
    noise = jax.random.normal(key, samples.shape)
    perturbed_samples = samples + noise * sigmas
    target = -noise / sigmas
    scores = model(perturbed_samples, labels)
    loss = 1 / 2. * ((scores - target) ** 2).sum(axis=-1) * sigmas.squeeze() ** anneal_power
    return loss.mean(axis=0)


# In[10]:


""" 
The training of the NCSNv2 model. Here define the training
parameters and initialise the model. Train on a small scale 
for testing before moving to the full scale on GPU HPC.
"""
# ----------- #
# model setup #
# ----------- #

# load in data  
box_size = 31
dataname = 'sources_box' + str(box_size) + '.npy'     
dataset = np.load(dataname)
#plt.imshow(dataset[2], cmap='gray')
#plt.show()

# perform zero-padding of the data to get desired dimensions
data_padded_31 = []
#dataset = np.resize(dataset,(1989,96,96))
for i in range(len(dataset)):
    data_padded_tmp = np.pad(dataset[i], ((0,1),(0,1)), 'constant')
    data_padded_31.append(data_padded_tmp)
dataset = np.array( data_padded_31 )

# define noise levels 
sigma_begin = 1
sigma_end   = 0.01
num_scales  = 10
sigmas      = np.exp(np.linspace(np.log(sigma_begin), 
                        np.log(sigma_end), num_scales))

# score model params
n_epochs    = 50                                    # number of epochs
steps       = 1_000                                 # number of steps per epoch
batch_size  = 32                                    # batch size
lr          = 1e-4                                  # learning rate
rng         = jax.random.PRNGKey(1992)              # random seed
input_shape = (jax.local_device_count(), 32, 32, 1) # size 32 by 32 one channel
label_shape = input_shape[:1]
fake_input  = jnp.zeros(input_shape)
fake_label  = jnp.zeros(label_shape, dtype=jnp.int32)
params_rng, dropout_rng = jax.random.split(rng)
model = NCSNv2()
#model = model_def()
variables = model.init({'params': params_rng}, fake_input, fake_label)
# Variables is a `flax.FrozenDict`. It is immutable and respects functional programming
init_model_state, initial_params = variables.pop('params')
optimizer = flax.optim.Adam(learning_rate=lr,
                            beta1 = 0.9,
                            eps = 1e-8).create(initial_params)  # create optimizer

# ------------- #
# training loop #
# ------------- #
@jax.jit
def train_step(model, optimizer, rng, samples, labels, sigmas):
    rng   = jax.random.PRNGKey(rng) # random number random seed
    grads = jax.grad(anneal_dsm_score_estimation)(model, samples, labels, sigmas, rng)
    model = optimizer.update(grads, model)
    return model, optimizer, rng

key_seq = jax.random.PRNGKey(0)
for t in tqdm(range(steps + 1)):

    idx = np.random.randint(0, len(dataset))
    labels = np.random.randint(0, len(sigmas), size=len(dataset[idx])) # size for 32 by 32
    model, optimizer, key_seq = train_step(model, optimizer, 
                                key_seq, dataset[idx], labels, sigmas[labels])

    if ((t % (steps // 5)) == 0):
        labels = np.random.randint(0, len(sigmas), size=len(dataset[0]))
        print(anneal_dsm_score_estimation(model, dataset[0], labels, sigmas[labels], rng))
# -------------------- #
# end of training loop #       
# -------------------- #


# In[ ]:


# ------------------------ #
# testing score estimation #
# ------------------------ #
gaussian_noise = jax.random.normal(rng, shape=(32,32))
galaxy = dataset[1992]
labels = np.random.randint(0, len(sigmas), (gaussian_noise.shape[0],))
scores = model(gaussian_noise, labels)
scores2 = model(galaxy, labels)
fig , ax = plt.subplots(1,2,figsize=(16, 5.5), facecolor='black',dpi = 70)
plt.subplots_adjust(wspace=0.01)
plt.subplot(1,2,1)
plt.imshow(scores, cmap='plasma')
#plt.colorbar()
plt.title('Gaussian Noise',fontsize=28,pad=15)
plt.subplot(1,2,2)
plt.imshow(scores2, cmap='plasma')
cbar = plt.colorbar()
cbar.set_label(r'$\nabla_x log \ p(\mathbf{\tilde{x}})$', rotation=270, fontsize = 20,labelpad= 25)
plt.title('Galaxy',fontsize=28,pad=15)


# In[ ]:


# ------------------------- #
# langevin dynamic sampling #
# ------------------------- #
# TODO: port to jax

def anneal_Langevin_dynamics(x_mod, scorenet, sigmas, n_steps_each=100, step_lr=0.000008,
                             final_only=False, verbose=False, denoise=True):
    images = []
    scores  = []

    for c, sigma in enumerate(sigmas):
        labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c
        labels = labels.long()
        step_size = step_lr * (sigma / sigmas[-1]) ** 2
        step_size_cpu = step_size.to('cpu') 
        for s in range(n_steps_each):
            grad = scorenet(x_mod, labels)
            scores.append(grad.to('cpu'))
            noise = torch.randn_like(x_mod)
            grad_norm = torch.norm(grad.view(grad.shape[0], -1), dim=-1).mean()
            noise_norm = torch.norm(noise.view(noise.shape[0], -1), dim=-1).mean()
            x_mod = x_mod + step_size_cpu * grad + noise * np.sqrt(step_size_cpu * 2)

            if not final_only:
                images.append(x_mod.to('cpu'))

    if denoise:
        last_noise = (len(sigmas) - 1) * torch.ones(x_mod.shape[0], device=x_mod.device)
        last_noise = last_noise.long()
        x_mod = x_mod + sigmas[-1] ** 2 * scorenet(x_mod, last_noise)
        images.append(x_mod.to('cpu'))

    return images, scores

