What is an Autoencoder? 

As described in the paper, an autoencoder is a neural network that consists of two parts:

1. The encoder, which compresses high dimensional data into low dimensions, capturing only the relevant information.

2. The decoder, which decodes the low dimensional back to the original dimension of the input and reconstructing the data by capturing much less features. 

There are a handful of types of autoencoders, namely: 
1. Sparse/Contractive
2. Denoising 
3. Convolutional
4. Variational 

Let's go over at these in a little more detail: 

1. Sparse/Contractive: To the usual autoencoder architecture, we will add a penalty/regularization term, mostly this is the weight decay which will prevent the neural network from overfitting. 

2. Denoising: The autoencoder will still caputre the most important features and "denoises" the dataset, we will add a weight decay term to further make the model more effective. 

3. Convolutional: Instead of using perceptrons aka linear layers, we will replace it with convolutional layers. The implementation for that is redundant and has not been added to this notebook.

4. Variational: Variational Autoencoders are the unique kind of the bunch, where we use this autoencoder for generative modelling. However, it should be noted tahat unlike a GAN, it cannot generate new information. 
    
    Unlike the usual autoencoder, VAEs learn from the underlying probability distribution of the latent space, and applying the latent variables to the encoded space, creating our decoded output. 

For more details, read the paper: [Autoencoders](https://arxiv.org/abs/2003.05991)