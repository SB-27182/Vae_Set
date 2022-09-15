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
Vae1 is a foundational model. It is uses a multivariate Gaussian as the latent distribution and either a bernoulli or gaussian reconstruction density depending on the data type. Vae1 is very vanilla in this regard.
<br>
<br>
Vae1 is designed to be a very robust and explicit VAE model. This explicit coding style is to allow for significant access to the inner workings of the model. This allows Vae1 to be quickly extended into novel experimental models. Below are some of the design choices that highlight this.
<br>
<br>
<br>

## `Design Choices: `

#### `Analysis specific models:`
After training a Vae1, the model state is saved. Analysis-specific variants of Vae1 are then able to load in this saved state. The analysis-specific Vae1 are well equipt to traverse the latent density using many parameterized, and manual algorithms. This encapsulation allows different types of analysis-specific extensions to be written apart from the training of the model.

#### `Dictionary Passing:`
Every component of Vae1 (encoder, sampling layer, etc) passes dictionary objects between each other. This allows extensions, where new objects need to be passed, to be easily appended to the basal objects being passed already.

#### `Encapsulated Probability Modules:`

