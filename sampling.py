#!/usr/bin/env python
# coding: utf-8
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
# import the Scorenet architecture
from model import ScoreNet
# test we can find correct device
from jax.lib import xla_bridge
import argparse

# Scorenet asci art


print('    _____                         _   _        _   ')
print('   /  ___|                       | \ | |      | |  ')
print('   \ `--.   ___  ___   _ __  ___ |  \| |  ___ | |_  ')
print('    `--. \ / __|/ _ \ |  __|/ _ \| . ` | / _ \| __|')
print('   /\__/ /| (__| (_) || |  |  __/| |\  ||  __/| |_ ')
print('   \____/  \___|\___/ |_|   \___|\_| \_/ \___| \__|')
print('   Generating galaxies from noise with deep learning')     
print('                  <>  Matt Sampson  <>')                                       

#print(f'Device used: {xla_bridge.get_backend().platform}')

# parse in the image size to train on from the command line
# Parse arguements
parser = argparse.ArgumentParser(
    description="training script")
parser.add_argument("-s", "--size",
                    help="size of image to train on",
                    default="32", type=int)
parser.add_argument("-t", "--epoch",
                    help="save param epoch",
                    default="32", type=int)
args    = parser.parse_args()


# ------------------------------------------------------------ #
# The training of the NCSNv2 model. Here define the training   #
# parameters and initialise the model. Train on a small scale  #
# for testing before moving to the full scale on GPU HPC.      #
# ------------------------------------------------------------ #

def createData(im_size):
    """
    Create the data set for training
    -------------------------------------------
    Inputs: box_size - the size of the box to create
    Output: dataset  - the data set
    -------------------------------------------
    """
    # create the data set
    if im_size == 32:
        box_size = 31
        dataname = 'sources_box' + str(box_size) + '.npy'     
        dataset = np.load(dataname)
        # perform zero-padding of the data to get desired dimensions
        data_padded_31 = []
        for i in range(len(dataset)):
            data_padded_tmp = np.pad(dataset[i], ((0,1),(0,1)), 'constant')
            data_padded_31.append(data_padded_tmp)
        dataset = np.array( data_padded_31 )
        
    elif im_size == 64:
        box_size = 51
        dataname = 'sources_box' + str(box_size) + '.npy'     
        dataset = np.load(dataname)
        # perform zero-padding of the data to get desired dimensions
        data_padded_51 = []
        for i in range(len(dataset)):
            data_padded_tmp = np.pad(dataset[i], ((6,7),(6,7)), 'constant')
            data_padded_51.append(data_padded_tmp)
        dataset_51 = np.array( data_padded_51 )
        # load in data  high res
        box_size = 61
        dataname = 'sources_box' + str(box_size) + '.npy'     
        dataset = np.load(dataname)
        # perform zero-padding of the data to get desired dimensions
        data_padded_61 = []
        for i in range(len(dataset)):
            data_padded_tmp = np.pad(dataset[i], ((1,2),(1,2)), 'constant')
            data_padded_61.append(data_padded_tmp)
        # add a loop to add 51 and 61 data together
        #for i in range(len(dataset_51)):
        #    data_padded_61.append( dataset_51[i] )
        dataset = np.array( data_padded_61 )

    # convert dataset to jax array
    dataset = np.expand_dims(dataset, axis=-1)
    data_jax = jnp.array(dataset)
    
    return data_jax

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


# ------------------- #
# model training step #
# ------------------- #

# model training and init params
key_seq       = jax.random.PRNGKey(42)               # random seed
n_epochs      = 20                                   # number of epochs
batch_size    = 1 #64                                   # batch size
lr            = 1e-4                                 # learning rate
im_size       = args.size                            # image size
training_data = createData(im_size)                  # create the training data

# construct the training data 
# for testing limit size until GPU HPC is available
len_train = 2500#80000
training_data = training_data[0:len_train] # DELETE for full training
batch = jnp.array(range(0, batch_size))
training_data_init = training_data[batch]
batch_per_epoch = len(training_data) // batch_size

# define noise levels and noise params
sigma_begin = 1
sigma_end   = 0.01
num_scales  = 10
sigmas      = jnp.exp(jnp.linspace(jnp.log(sigma_begin), 
                        jnp.log(sigma_end),num_scales))
labels = jax.random.randint(key_seq, (len(training_data_init),), 
                            minval=0, maxval=len(sigmas), dtype=jnp.int32)

# model init variables
input_shape = training_data_init.shape
label_shape = labels.shape
fake_input  = jnp.zeros(input_shape)
fake_label  = jnp.zeros(label_shape, dtype=jnp.int32)
params_rng, dropout_rng = jax.random.split(key_seq)

# define and initialise model
model = ScoreNet()
# JIT compile the model for faster training
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

# training settings
CKPT_DIR    = 'saved_params_' + str(im_size)

# ------------------------- #
# langevin dynamic sampling #
# ------------------------- #
def anneal_Langevin_dynamics(x_mod, scorenet, best_params, sigmas, rng, n_steps_each=100, 
                                step_lr=0.000008,denoise=True):
    # initialise arrays for images and scores
    images = []
    scores = []

    # loop over noise levels from high to low for sample generation
    for c, sigma in enumerate(sigmas):
        labels = jax.numpy.ones(x_mod.shape[0],dtype=np.int8) * c
        step_size = step_lr * (sigma / sigmas[-1]) ** 2
        desc = 'sampling at noise level: ' + str(c + 1) + ' / ' + str(len(sigmas))
        for s in tqdm(range(n_steps_each),desc=desc):
            grad = scorenet.apply({'params' : best_params}, x_mod, labels)
            noise = jax.random.normal(rng, shape=x_mod.shape)
            x_mod = x_mod + step_size * grad + noise * jax.numpy.sqrt(step_size * 2)
            images.append(x_mod[0].squeeze())
            scores.append(grad[0].squeeze())

    # add a final denoising step is desired
    if denoise:
        last_noise = (len(sigmas) - 1) * jax.numpy.ones(x_mod.shape[0], dtype=np.int8)
        last_grad = scorenet.apply({'params' : best_params}, x_mod, last_noise)
        x_mod = x_mod + sigmas[-1] ** 2 * last_grad
        images.append(x_mod[0].squeeze())
        scores.append(last_grad[0].squeeze())

    return images, scores

# TODO: JIT compile the function for serious speed ups in the for loop
# think will need static model args - use functool.partial(XXXX)?
#anneal_Langevin_dynamics = jax.jit(anneal_Langevin_dynamics)

# ---------------- #
# testing sampling #
# ---------------- #
n_samples      = 1                               # number of samples to generate
sample_steps   = 100                              # number of steps to take at each noise level
shape_array    = jnp.array(range(0, n_samples))  # run Langevin dynamics on n_samples
data_shape     = training_data[shape_array]           # get the data shape for starting image
gaussian_noise = jax.random.normal(key_seq, shape=data_shape.shape) # Initial noise image/data

# load best model 
# TODO: clean this up and comment
scorenet = model # nicer name for the model
filename = CKPT_DIR + '/scorenet_' + str(im_size) + '_epoch_' + str(args.epoch) + '.pickle'
# load the weights and biases with pickle
with open(filename, 'rb') as handle:
    best_params = pickle.load(handle)
    
# run the Langevin sampler
images, scores = anneal_Langevin_dynamics(  gaussian_noise, 
                                            scorenet,
                                            best_params, 
                                            sigmas, 
                                            key_seq,
                                            n_steps_each=sample_steps, 
                                            denoise=True  )

# ------------------------------------------- #
# plot sample evolution with generation steps #
# ------------------------------------------- #
# NOTE: for single sample data, easy to adjust if running multiple samples

images_array = np.array(images)
col_map = cmr.lilac
fig , ax = plt.subplots(2,5,figsize=(16, 7), facecolor='white',dpi = 70)
plt_idx = int( len(images_array) / 10 )
n_panels = 7
step_array =  range(0, sample_steps, int(sample_steps / (n_panels**2 - 1)) )
for i in range(n_panels):
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.subplot(n_panels,n_panels,i + 1)
    if (i < n_panels - 1):
        step = step_array[i] * num_scales # note num_scales factor is for the number of noise levels
        name = 'step ' + str(int( step / num_scales ))
        plt.title(name, fontsize = 24)
        plt.imshow(images_array[step], cmap=col_map)
        plt.axis('off')
    else: 
        name = 'final step ' + str(int( sample_steps ))
        plt.title(name, fontsize = 24)
        plt.imshow(images_array[-1], cmap=col_map)
        plt.axis('off')
plt.tight_layout()
name = 'langevin_sampling_panels_res' + str(im_size) + '.png'
plt.savefig(name,facecolor='white',dpi=300)
plt.show()

