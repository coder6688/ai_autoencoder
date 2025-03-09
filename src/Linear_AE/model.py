import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearAutoencoder(nn.Module):
    '''
    Input img -> Hidden dim -> mean, std -> Reparameterization(z) -> Decoder: Hidden dim -> Output img
    '''
    def __init__(self, input_dim, hid1_dim=128, hid2_dim=64, hid3_dim=16, hid4_dim=3):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hid1_dim),
            nn.ReLU(),
            nn.Linear(hid1_dim, hid2_dim),
            nn.ReLU(),
            nn.Linear(hid2_dim, hid3_dim),
            nn.ReLU(),
            nn.Linear(hid3_dim, hid4_dim)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hid4_dim, hid3_dim),
            nn.ReLU(),
            nn.Linear(hid3_dim, hid2_dim),
            nn.ReLU(),
            nn.Linear(hid2_dim, hid1_dim),
            nn.ReLU(),
            nn.Linear(hid1_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):   
        z = self.encoder(x)
        x = self.decoder(z)
        return x, z
    
    def decode(self, z):
        return self.decoder(z)


# import torch
if __name__ == '__main__':
    torch.manual_seed(42)

    x = torch.randn(1, 28*28) # 1 batch, 28*28 image

    autoencoder = LinearAutoencoder(input_dim=784, hid1_dim=128, hid2_dim=64, hid3_dim=16, hid4_dim=2)
    print(autoencoder)

    print(f'before: {x.shape}')
    x = autoencoder(x)

    x = x.view(1, 28, 28)
    print(f'after: {x.shape}')
