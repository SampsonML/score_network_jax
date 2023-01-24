## Score-matching neural networks for likelihood based galaxy morphology priors
### Matt Sampson

For the use of galaxy deblending we construct a time-independent 
score matching model (diffusion model) with a U-NET architecture. The architecture
is a JAX port from Song+21 NCSN (https://arxiv.org/abs/2006.09011).

This score network is to be used as a galaxy morphology prior in the upcoming 
SCARLET 2 code (2 fast 2 blurry-less). 

DETAILS:
The two main scripts are scorenet_32.py and scorenet_64.py which are
the training scripts for the 32 by 32 and 64 by 64 resolution scarlet
models respectively. A template jobscript for use on the Princeton HPC Della
is added. Training states will be auto-saved to ckpt_32 and ckpt_64 directories
inline with the two different resolution trials.

For details/issues/plot aesthetic suggestions
email: matt.sampson@princeton.edu 
