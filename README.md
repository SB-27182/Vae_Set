## `Vae-Set:` &nbsp; A collection of expiramental VAEs built on a robust foundational model

<p align="center">
  <kbd>
  <img src="https://github.com/SB-27182/Vae_Set/blob/master/readme_images/topOfSeven.jpg" width=700 height=77 />
  </kbd>
  <br>
  <sub>For an example of an experimental model see Vae4 (below)</sub> 
</p>
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


#### `Dictionary Passing:`
Every component of Vae1 (encoder, sampling layer, etc) passes dictionary objects between each other. This allows extensions, wherein new objects need to be passed, to be built easily.
<br>

#### `Explicit Probability Densities:`
The latent probability layers/likekihoods are written explicity. This reveals the simple laws a variational auto encoder must abide by. We see that the heart of the VAE, the multivariate gaussian density, contains observable elements that can be added, and scaled, while still being inside the gaussian density. This is to say, the latent density is closed under addition and scalar multiplication; it also has a measure of magnitude. Indeed, the multivariate gaussian is a Hilbert space.
<p align="center">
  <kbd>
  <img src="https://github.com/SB-27182/Vae_Set/blob/master/readme_images/explicit_probs.png" width=480 height=268 />
  </kbd>
</p>
Disentanglement/principal-component-analysis, are the consequences of this orthogonal Hilbert space. Naturally, this opens the door for many new theoretical modifications of the VAE. Another Hilbert space that has caused revolutionary developments in NLP is the Fourier space. Indeed, attention-based/transformer models use a frequency basis to encode latent signals of sequential data. In this regard, there is a shared core structure between transformers and VAEs. It seems hybrid models are possible. 
<br>

#### `Analysis specific models:`
After training a Vae1, the model state is saved. Analysis-specific variants of Vae1 are then able to load in this saved state. The analysis-specific Vae1 variants are well equipt to traverse the latent density using many parameterized, and manual algorithms. This encapsulation allows different types of analysis-specific extensions to be written apart from the training of the model.
<br>
<br>
<br>
<br>

# `Vae4:` &nbsp; A Gaussian-Categorical-Joint Density Model
Data is often recognized as existing in some high dimensional manifold. This manifold represents the behavior of the underlying generator function, and thus ML architectures like invertible flow networks, can learn these hyper-dimensional smooth structures. <br>
However in real life, there are often many "states" that these generator functions can be in. In the study of dynamical systems, these "states" are usually intuited as different parameterizations of the underlying generator function, following bifurcation events. 
(But that's neither here nor there, the point is these "states" exist in data).
<br>
<br>
To a geometer, these would be described as "discrete structures" in the manifold. To a statistician, the data would be described as multi-modal, or "clustered". At anyrate, Vae4 uses a categorical-multivariate gaussian-joint density as the latent probability distribution to  <ins>**learn this manifold without the use of any labels.**</ins>
<br>
<br>

## `Example Application`
<p align="center">
  <kbd>
  <img src="https://github.com/SB-27182/Vae_Set/blob/master/readme_images/anomalous1.png" width=179 height=200 />
  </kbd>
</p>
Although applications of this model are innumerable, here is an example concerning anomaly detection.
Suppose we obtain an anomalous observation from nature. (In reality, this can be either a set of data or a single observation.)
<br>
<br>
<br>
<p align="center">
  <kbd>
  <img src="https://github.com/SB-27182/Vae_Set/blob/master/readme_images/categorical1.png" width=500 height=182 />
  </kbd>
</p>
We want to know what this value should be categorized as. <ins>There is very little overlap of the anomaly's discrete-signal and the 2-discrete-signal.</ins> Naturaly, Vae4 does not categorize the anomalous observation as a 2.
<br>
Vae4 categorizes the value as an element in the 5-cluster. Suppose we would like to dig deeper, we want to see how the anomalous-5 compares to a very probable 5, generated from the mean latent dimensions of the 5 cluster.
<br>
<br>
<br>
<br>
<p align="center">
  <kbd>
  <img src="https://github.com/SB-27182/Vae_Set/blob/master/readme_images/similarity1.png" width=500 height=241 />
  </kbd>
</p>
On dimensions 49 and 39 we see that the anomalous-5 and the mean-5 are very similar. Vae4 is saying that the anomalous-5 is very probable with respect to its general width, and the width of its line.
<br>
<br>
<br>
<br>
<p align="center">
  <kbd>
  <img src="https://github.com/SB-27182/Vae_Set/blob/master/readme_images/difference1.png" width=500 height=241 />
  </kbd>
</p>
However, we see here what the issue with the anomalous-5 actually is. Vae4 is saying that the top horizontal line of the 5, is abnormally long (dimension 34). Vae4 is also saying that the lower tail of the anomalous-5 is abnormally small (dimension 44).

<br>
<br>

 <sub>**Note: Vae2-Vae4 are not in this repo at this time. Some of these expiramental models may be publishable.**</sub> 
