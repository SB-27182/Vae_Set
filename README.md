## `Vae-Set:` &nbsp; A collection of experimental VAEs built on a robust foundational model

<p align="center">
  <kbd>
  <img src="https://github.com/SB-27182/Vae_Set/blob/master/readme_images/topOfSeven.jpg" width=700 height=77 />
  </kbd>
  <br>
  <sub>Here is a latent structure of about 60 variables, found to exist within the human handwriting of the number 7.<br> This is a normally distributed feature. <br> See Vae4 (below)</sub> 
</p>
<br>
<br>


## `Vae1:` &nbsp;
Vae1 is a foundational model. It uses a multivariate Gaussian as the latent distribution and either a continuous-bernoulli or gaussian reconstruction density depending on the data type.
<br>
<br>
It is designed to be a very robust and explicit VAE model. This explicit coding style allows for significant access to the inner workings of the model. Thus, Vae1 can be quickly extended into novel experimental architectures. Below are some of the design choices that make this possible.
<br>
<br>
<br>

## `Design Choices: `


#### `Dictionary Passing:`
Every component of Vae1 (encoder, sampling layer, etc) passes dictionary objects between each other. This allows extensions, wherein new objects need to be passed, to be easily appended.
<br>

#### `Explicit Probability Densities:`
The latent probability layers/likelihoods are written explicitly. This reveals the simple laws a variational auto encoder must abide by. We see that the heart of most VAE models, the family of multivariate gaussian densities, contains parameterized elements that can be added and scaled, while still being inside the span of possible multivariate-Gaussian densities. This is to say, the space of parameterized latent density functions is closed under addition and scalar multiplication. Indeed, parameterized multivariate gaussians describe a mathematical Vector space.
<p align="center">
  <kbd>
  <img src="https://github.com/SB-27182/Vae_Set/blob/master/readme_images/explicit_probs.png" width=480 height=268 />
  </kbd>
</p>
Disentanglement/Independent-Component-Analysis in this setting, are the consequences of an orthogonal vector basis. Naturally, this opens the door for many new theoretical modifications of the VAE. Another generalized vector space that has caused revolutionary developments in NLP is the Fourier space. Indeed, attention-based/transformer models use a frequency basis to encode latent signals of sequential data. There is a shared, mathematical core structure between transformers and VAEs. It seems hybrid models are possible. 
<br>

#### `Analysis specific models:`
After training a Vae1, the model state is saved. Analysis-specific variants of Vae1 are then able to load in this saved state. The analysis-specific Vae1 variants are well equipped to traverse the latent density using many parameterized, and manual algorithms. This encapsulation allows different types of analysis-specific extensions to be written apart from the training of the model.
<br>
<br>
<br>
<br>

# `Vae4:` &nbsp; A Gaussian-Categorical-Joint Density Model
Data is often recognized as existing in some high dimensional manifold. This manifold represents the behavior of the underlying generator function, and thus ML architectures like invertible flow networks, can learn these hyper-dimensional smooth structures. <br>
However in real life, there are often many "states" that these generator functions can be in. In the study of dynamical systems, these "states" are usually intuited as different parameterizations of the underlying generator function (usually due to an unseen bifurcation event). 
But that's neither here nor there, the point is these "states" exist in data.
<br>
<br>
To a geometer, these would be described as "discrete structures" in the manifold. To a statistician, the data would be described as multi-modal, or "clustered". At anyrate, Vae4 uses a categorical, multivariate gaussian joint-density as the latent probability distribution to  <ins>**learn this manifold without the use of any labels.**</ins>
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
We want to know what this value should be categorized as. As can be seen <ins>there is very little overlap of the anomaly's <i>discrete-signal</i> and the <i>2-discrete-signal</i></ins>. Naturaly, Vae4 does not categorize the anomalous observation as a <b>2</b>.
<br>
Vae4 categorizes the value as an element in the <b>5-cluster</b>. To answer the question of <i><b>why</b></i> the anomalous data is categorized as a <b>5</b>, we may investigate how it compares to a <b>known-5</b> that is already an element of the <b>5-cluster</b>.
<br>
<br>
<br>
<br>
<p align="center">
  <kbd>
  <img src="https://github.com/SB-27182/Vae_Set/blob/master/readme_images/similarity1.png" width=500 height=241 />
  </kbd>
</p>
On dimensions 49 and 39 we see that the <b>anomalous-5</b> and the <b>known-5</b> are very similar. Vae4 is saying that the <b>anomalous-5</b>, with respect to its "<i>over-all width</i>", as well as it's "<i>average width of line</i>" could have very probably been generated from the <b><i>standard-5 parameterization</i></b> distribution. This is in the context of hypothesis testing, naturally.
<br>
<br>
<br>
<br>
<p align="center">
  <kbd>
  <img src="https://github.com/SB-27182/Vae_Set/blob/master/readme_images/difference1.png" width=500 height=241 />
  </kbd>
</p>
However, we see here what the particularities of the <b>anomalous-5</b> actually are. Vae4 is saying that the "<i>top horizontal line length</i>" of the <b>anomalous-5</b>, has an abnormally large value (dimension 34). Vae4 is also saying that the <b>anomalous-5</b> has a "<i>lower-loop to upper-loop size ratio</i>" that is abnormally in favor of the "<i>upper-loop</i>" (dimension 44).

<br>
<br>

 <sub>**Note: Vae2,3,4 are not in this repo at this time. Some of these expiramental models may be publishable.**</sub> 
