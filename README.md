This repo contains the implementation of Linear Autoencoder, Convolutional Autoencoder, and Variational Autoencoder. The Linear Autoencoder is implemented using a linear neural network with ReLU activation functions, while the Convolutional Autoencoder uses convolutional layers with ReLU activation functions. The Variational Autoencoder uses a Gaussian prior distribution for the latent variables.

These Autoencoders are trained on the MNIST dataset. The results are shown below.

It is interesting to see that the loss functions have dramatically different performances for VAE.


## Autoencoder Linear NN
After epcho=10

![Linear Autoencoder Results](resources/linear_ae_epoch_10.png)

After epcho=50

![Linear Autoencoder Results](resources/linear_ae_epoch_50.png)


## Autoencoder Convolutional NN
After epcho=10

![Convolutional Autoencoder Results](resources/cae_epoch_10.png)

After epcho=50

![Convolutional Autoencoder Results](resources/cae_epoch_50.png)

## Variational Autoencoder NN
### Large performance difference between loss functions of BCELoss and MSELoss
#### criterion = BCELoss(reduction='sum')
After epcho=10

![Variational Autoencoder Results](resources/vae_epoch_BCELoss_10.png)

After epcho=20

![Variational Autoencoder Results](resources/vae_epoch_BCELoss_20.png)

After epcho=50

![Variational Autoencoder Results](resources/vae_epoch_BCELoss_50.png)

#### criterion = MSELoss()
After epcho=10

![Variational Autoencoder Results](resources/vae_epoch_MSELoss_10.png)

After epcho=20

![Variational Autoencoder Results](resources/vae_epoch_MSELoss_20.png)

After epcho=50

![Variational Autoencoder Results](resources/vae_epoch_MSELoss_50.png)

### References
```
Auto-Encoding Variational Bayes, Diederik P Kingma, Max Welling
https://arxiv.org/abs/1312.6114
```
