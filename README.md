# Score-matching neural networks for likelihood based galaxy morphology priors
## Matt Sampson

UPDATE - moved to a model using equinox

For the use of galaxy deblending we construct a time-independent 
score matching model (diffusion model) with a U-NET architecture. The architecture
is a JAX port from Song+21 NCSN (https://arxiv.org/abs/2006.09011).

This score network is to be used as a galaxy morphology prior in the upcoming 
SCARLET 2 details: https://pmelchior.net/blog/scarlet2-redesign.html. Scarlet 2 code here: https://github.com/pmelchior/scarlet2

## Details:
The neural network architecture "ScoreNet" is located in models.py
The loss function is defined in the training script train_script.py which takes in command
line inputs for the image size to train on. 
Parameters are saved via pickling files and simply loading them in when needed.
A template jobscript for use on the Princeton HPC Della
is added. Training states will be auto-saved to params_32 and params_64 directories
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
