# Score-matching neural networks for likelihood based galaxy morphology priors
## Matt Sampson

For the use of galaxy deblending we construct a time-independent 
score matching model (diffusion model) with a U-NET architecture. The architecture
is a JAX port from Song+21 NCSN (https://arxiv.org/abs/2006.09011).

This score network is to be used as a galaxy morphology prior in the upcoming 
SCARLET 2 code (https://pmelchior.net/blog/scarlet2-redesign.html) 2 fast 2 blurry-less. 

## Details:
The two main scripts are scorenet_32.py and scorenet_64.py which are
the training scripts for the 32 by 32 and 64 by 64 resolution scarlet
models respectively. A template jobscript for use on the Princeton HPC Della
is added. Training states will be auto-saved to ckpt_32 and ckpt_64 directories
inline with the two different resolution trials.

## Useful papers
### For context scientific context:

Scarlet paper: (https://ui.adsabs.harvard.edu/abs/2018A&C....24..129M)

### Similar work:

Song+2019 and 2020 (https://arxiv.org/abs/1907.05600 , https://arxiv.org/abs/2006.09011)

Burke+2019 (https://arxiv.org/abs/1908.02748)

Huang+2022 (https://www.assemblyai.com/blog/diffusion-models-for-machine-learning-introduction/)

## Useful other rescources
Lilian Weng blogpost (https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)

Yang Song blogpost (https://yang-song.github.io/blog/2021/score/)

For details/issues/plot aesthetic suggestions
email: matt.sampson@princeton.edu 
