#!/usr/bin/env python
# coding: utf-8
# ------------------------------------------------------------------------------ #
# A jax implementation of the diffusion model from                               #
# the paper "Improved Techniques for Training Score-Based Generative Models"     #
# https://arxiv.org/abs/2006.09011                                               #
# Code taken primarily from https://github.com/yang-song/score_sde/              #
# Author: Matt Sampson                                                           #
# Created: 2023                                                                  #
#                                                                                #
# Main modifications by Matt Sampson include:                                    #
#    - Minor updates to U-NET use the latest version of JAX/flax.linen           #
#    - removal of config files all params define in python file                  #
#    - data loader added using numpy arrays then converted to jax arrays         #
#    - changed optim.Adam to optax.adam (required for latex flax)                # 
#    - re-wrote optimisation routine to use optax                                #
#    - replaced the training loop with mini-batch grad descent in optax          #
#    - re-wrote Langevin sampler in JAX soon with JIT (much faster sampling)     #
#    - addition of various visualisation routines                                #
# ------------------------------------------------------------------------------ #

import os
import pickle
import functools
import math
import string
from typing import Any, Sequence, Optional
import flax
import flax.linen as nn
import optax
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.nn.initializers as init
from jax import jit
import matplotlib as mpl
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


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

# Changed bias to false
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
                   kernel_init=kernel_init)(x)
                   #bias_init=bias_init)(x)
  return output

# --------------------------------------------------------- #
# Functions below are ported over from the NCSNv2 codebase: #
#        https://github.com/ermongroup/ncsnv2               #
# --------------------------------------------------------- #

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


# --------------------------------------------------------------------------- #
# JAX port of ncsnv2 - updated to work with                                   #
# latest JAX/FLAX versions using optax over flax.optim                        #
# original: https://github.com/yang-song/score_sde/blob/main/models/ncsnv2.py #
# --------------------------------------------------------------------------- #
CondResidualBlock = ConditionalResidualBlock
conv3x3 = ncsn_conv3x3

class ScoreNet(nn.Module):
  """ScoreNet model architecture"""
  #config: ml_collections.ConfigDict

  @nn.compact
  #def __call__(self, x, labels, train=True):
  def __call__(self, x, labels):
    
    # hard coding configs for now
    sigma_begin   = 1                     # noise scale max
    sigma_end     = 1e-2                  # noise scale min
    num_scales    = 10                     # number of noise scales
    sigmas        = jnp.exp(jnp.linspace(
                                jnp.log(sigma_begin), 
                                jnp.log(sigma_end), num_scales))
    nf            = 128                   # number of filters
    act           = nn.elu                # activation function
    normalizer    = InstanceNorm2dPlus    # normalization function
    interpolation = 'bilinear'            # interpolation method for upsample
    data_centered = False                 # whether data is already centered
    
    # data already centered
    if not data_centered:
        h = 2 * x - 1.
    else:
        h = x
    
    # Begin the U-Net
    h = conv3x3(h, nf, stride=1, bias=True)
    
    # TODO: check the size of nf is correct

    # ResNet backbone
    h = ResidualBlock(nf, resample=None, act=act, normalization=normalizer)(h)
    layer1 = ResidualBlock(nf, resample=None, act=act, normalization=normalizer)(h)
    h = ResidualBlock(2 * nf, resample='down', act=act, normalization=normalizer)(layer1)
    layer2 = ResidualBlock(2 * nf, resample=None, act=act, normalization=normalizer)(h)
    h = ResidualBlock(  2 * nf,
                        resample='down',
                        act=act,
                        normalization=normalizer,
                        dilation=2)(layer2)
    layer3 = ResidualBlock(2 * nf, resample=None, act=act, normalization=normalizer, dilation=2)(h)
    h = ResidualBlock(  2 * nf,
                        resample='down',
                        act=act,
                        normalization=normalizer,
                        dilation=4)(layer3)
    layer4 = ResidualBlock(2 * nf, resample=None, act=act, normalization=normalizer, dilation=4)(h)
    # U-Net with RefineBlocks
    ref1 = RefineBlock( layer4.shape[1:3],
                         2 * nf,
                        act=act,
                        interpolation=interpolation,
                        start=True)([layer4])
    ref2 = RefineBlock(  layer3.shape[1:3],
                        2 * nf,
                        interpolation=interpolation,
                        act=act)([layer3, ref1])
    ref3 = RefineBlock( layer2.shape[1:3],
                         2 * nf,
                        interpolation=interpolation,
                        act=act)([layer2, ref2])
    ref4 = RefineBlock( layer1.shape[1:3],
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
    
    h = h / used_sigmas
    
    return h 