import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from model import Convolutional_Autoencoder
from util import device

torch.manual_seed(42)

OUTPUT_DIR = 'output/cae'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# configuration
INPUT_DIM = 784
HID1_DIM = 128
HID2_DIM = 64
HID3_DIM = 16
HID4_DIM = 3
LEARNING_RATE = 1e-4 #KARPATHY CONSTANT
BATCH_SIZE = 64
NUM_EPOCHS = 10

# dataset loading
dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
model = Convolutional_Autoencoder().to(device)

# loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

def save_image(imgs, fig_save_path, title, reshape=(-1, 28, 28)):
    for i, item in enumerate(imgs):
        if i >= 9: break
        plt.subplot(2, 9, i+1)
        item = item.reshape(reshape)
        plt.imshow(item[0])
        plt.axis('off')
        if i == 0:
            plt.title(title, pad=10) 

    plt.savefig(fig_save_path, bbox_inches='tight', dpi=300)
    plt.close()  # Close the figure to free memory

# training loop
for epoch in range(NUM_EPOCHS):
    for batch_data, _ in tqdm(train_loader):

        x = batch_data.view(-1, 1, 28, 28).to(device)
        x_reconstructed, z_latent = model(x)

        # reconstruction loss
        loss = criterion(x_reconstructed, x)

        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {loss.item():.4f}')

    plt.figure(figsize=(15, 4))
    plt.gray()

    # original images
    x = x.cpu().detach().numpy()
    np.save(os.path.join(OUTPUT_DIR, f'original_epoch_{epoch+1}.npy') , x.squeeze())
    save_image(x, os.path.join(OUTPUT_DIR, f'original_epoch_{epoch+1}.png'), 'Original')

    # reconstructed images
    imgs = x_reconstructed.cpu().detach().numpy()
    save_image(imgs, os.path.join(OUTPUT_DIR, f'reconstruction_epoch_{epoch+1}.png'), 'Reconstructed')

    # save the z_latent image and vector to files
    z_latent = z_latent.view(-1, 1, 8, 8).detach().cpu().numpy()
    save_image(z_latent, os.path.join(OUTPUT_DIR, f'latent_epoch_{epoch+1}.png'), 'Latent', reshape=(-1, 8, 8))
    np.save(os.path.join(OUTPUT_DIR, f'latent_epoch_{epoch+1}.npy') , z_latent.squeeze())

# save the model
torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f'cae_{epoch+1}.pth'))
