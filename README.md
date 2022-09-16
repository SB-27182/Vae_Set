## `Vae-Set:` &nbsp; A collection of expiramental VAEs built on a robust foundational model


<p align="center">
  <kbd>
  <img src="https://github.com/SB-27182/Vae_Set/blob/master/assets/readme_images/topOfSeven.jpg" width=700 height=77 />
  </kbd>
  <br>
  <sub>For an example of an experimental model see Vae4 (below)</sub> 
</p>
<br>
<br>
<br>

## `Vae1:` &nbsp;
Vae1 is a foundational model. It is uses a multivariate Gaussian as the latent distribution and either a bernoulli or gaussian reconstruction density depending on the data type.
<br>
<br>
It is designed to be a very robust and explicit VAE model. This explicit coding style allows for significant access to the inner workings of the model. Thus, Vae1 can be quickly extended into novel experimental architectures. Below are some of the design choices that make this possible.
<br>
<br>
<br>

## `Design Choices: `

#### `Analysis specific models:`
After training a Vae1, the model state is saved. Analysis-specific variants of Vae1 are then able to load in this saved state. The analysis-specific Vae1 variants are well equipt to traverse the latent density using many parameterized, and manual algorithms. This encapsulation allows different types of analysis-specific extensions to be written apart from the training of the model.

#### `Dictionary Passing:`
Every component of Vae1 (encoder, sampling layer, etc) passes dictionary objects between each other. This allows extensions, wherein new objects need to be passed, to be built easily.

#### `Explicit Probability Densities:`
The latent probability layers/likekihoods are written explicity. This reveals the simple laws a variational auto encoder must abide by. We see that the heart of the VAE, the multivariate gaussian density, contains observable elements that can be added, and scaled, while still being inside the gaussian density. This is to say, the latent density is closed under addition and scalar multiplication. Indeed, the multivariate gaussian is a Hilbert space.
<br>
<br>
Disentanglement/principal-component-analysis, are the consequences of this orthogonal Hilbert space. Naturally, this opens the door for many new theoretical modifications of the VAE. Another space that has caused revolutionary developments in NLP is the Fourier space. Indeed, attention-based/transformer models use a frequency basis to encode latent signals of sequential data.
<br>
<br>
<br>

# `Vae4:` &nbsp; A Multivariate Gaussian - Categorical - Joint Density Model
Data is often recognized as existing in some high dimensional manifold. This manifold represents the behavior of the underlying generator function, and thus ML architectures like invertible flow networks, can learn these hyper dimensional smooth structures. <br>
However in real life, there are often many "states" that these generator functions can be in. In the study of dynamical systems, these "states" are usually intuited as different parameterizations of the underlying generator function, following bifurcation events. 
(But that's neither here nor there, the point is these "states" exist in data).
<br>
<br>
To a geometer, these would be described as "discrete structures" in the manifold. To a statistician, the data would be described as multi-modal, or "clusters". At anyrate, Vae4 uses a categorical - multivariate gaussian - joint density as the latent probability distribution to  **learn this manifold without the use of any labels.**
