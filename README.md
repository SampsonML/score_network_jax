## Score network repository for a jax port of the NCSNv2 by Song+21
### Matt Sampson

For the use of galaxy deblending we construct a time-independent 
score matching model (diffusion model) with a U-NET architecture. The architecture
is a JAX port from Song+21 NCSN (https://arxiv.org/abs/2006.09011).

This score network is to be used as a galaxy morphology prior in the upcoming 
SCARLET 2 code (2 fast 2 blurry-less). 
