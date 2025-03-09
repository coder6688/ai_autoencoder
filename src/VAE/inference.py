import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from PIL import Image
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from model import VAE
from util import device

OUTPUT_DIR = 'output/vae'

BATCH_SIZE = 9
INPUT_DIM = 784
HID_DIM = 200
Z_DIM = 20

# Define the transformation to convert images to tensors    
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load the model
model = VAE(input_dim=INPUT_DIM, hid_dim=HID_DIM, z_dim=Z_DIM).to(device)  
criterion_name = 'MSELoss' # 'MSELoss' #'BCELoss'
EPOCH = 50
model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, f'vae_{criterion_name}_epoch_{EPOCH}.pth')))
model.eval()    

# Load the image
dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
for (img, _) in data_loader:
    break
# Process the batch
img = img.view(-1, INPUT_DIM).to(device)  # Flatten to (BATCH_SIZE, 784)

with torch.no_grad():
    x_reconstructed, mu, std = model(img)
    x_reconstructed = x_reconstructed.view(BATCH_SIZE, 28, 28).cpu().numpy()

    epsilon = torch.randn_like(std)
    z = mu + epsilon * std
    generated_img = model.decode(z).view(-1, 28, 28).cpu().numpy()
    z_img = z.view(-1, 4, 5).cpu().numpy()

    # Plot the images in a grid
    fig = plt.figure(figsize=(15, 6))

    # Plot original images
    for i in range(BATCH_SIZE):
        plt.subplot(3, 9, i + 1)
        plt.imshow(img[i].view(28, 28).cpu().numpy(), cmap='gray')
        plt.axis('off')
        if i == 0:
            plt.title('Original')

    # Plot generated images
    for i in range(BATCH_SIZE):
        plt.subplot(3, 9, i + 10)
        plt.imshow(z_img[i], cmap='gray')  # Take first image from batch
        plt.axis('off')
        if i == 0:
            plt.title('Latent')

    # Plot reconstructed images
    for i in range(BATCH_SIZE):
        plt.subplot(3, 9, i + 19)
        plt.imshow(x_reconstructed[i], cmap='gray')
        plt.axis('off')
        if i == 0:
            plt.title('Reconstructed')

    plt.tight_layout()
    plt.show()