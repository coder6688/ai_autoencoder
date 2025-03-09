import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    '''
    Input img -> Hidden dim -> mean, std -> Reparameterization(z) -> Decoder: Hidden dim -> Output img
    '''
    def __init__(self, input_dim, hid_dim=200, z_dim=20):
        super(VAE, self).__init__()

        # Encoder
        self.img_to_hid = nn.Linear(input_dim, hid_dim)
        self.hid_to_mean = nn.Linear(hid_dim, z_dim)
        self.hid_to_std = nn.Linear(hid_dim, z_dim)

        # Decoder
        self.z_to_hid = nn.Linear(z_dim, hid_dim) # reparameterization
        self.hid_to_img = nn.Linear(hid_dim, input_dim)

        # ReLU
        self.relu = nn.ReLU()

        # Sigmoid
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h = self.relu(self.img_to_hid(x))
        mean = self.hid_to_mean(h)
        std = self.hid_to_std(h)
        return mean, std

    def decode(self, z):
        h = self.relu(self.z_to_hid(z))
        return self.sigmoid(self.hid_to_img(h))

    def forward(self, x):   
        mu, std = self.encode(x)
        epislion = torch.randn_like(std)
        z_reparameterized = mu + epislion * std
        x_reconstructed = self.decode(z_reparameterized)
        return x_reconstructed, mu, std # return mu, std for KL divergence loss calculation

if __name__ == '__main__':
    torch.manual_seed(42)

    x = torch.randn(1, 28*28) # 1 batch, 28*28 image

    vae = VAE(input_dim=784)
    print(vae)

    x_reconstructed, mu, std = vae(x)
    print(x_reconstructed.shape)
    print(mu.shape)
    print(std.shape)
    print(mu, std)