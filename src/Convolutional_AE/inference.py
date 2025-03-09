import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import os
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from model import Convolutional_Autoencoder
import numpy as np
from util import device

OUTPUT_DIR = 'output/cae'

INPUT_DIM = 784
HID1_DIM = 128
HID2_DIM = 64
HID3_DIM = 16
HID4_DIM = 3
LEARNING_RATE = 1e-4 #KARPATHY CONSTANT
BATCH_SIZE = 64
NUM_EPOCHS = 10

# Define the transformation to convert images to tensors    
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load the model
model = Convolutional_Autoencoder().to(device)
model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, f'cae_{NUM_EPOCHS}.pth')))
model.eval()    

# Load the latent vector
epoch = 10
z_img = np.load(f'output/cae/latent_epoch_{epoch}.npy')
z_img = torch.from_numpy(z_img).float().view(-1, 64, 1, 1).to(device)

x_img = np.load(f'output/cae/original_epoch_{epoch}.npy')
x_img = torch.from_numpy(x_img).float().view(-1, 28, 28).to(device)


# Generate the image    
with torch.no_grad():
    x_reconstructed = model.decoder(z_img)

    plt.figure(figsize=(15, 6))  # Increased height for 3 rows
    plt.gray()
    
    # First row: Original loaded image (single column)
    imgs = x_img.cpu().detach().numpy()
    for i, item in enumerate(imgs):
        if i >= 9: break
        plt.subplot(3, 9, i+1)  # Position 1-9 for first row
        item = item.reshape(-1, 28, 28)
        plt.imshow(item[0])
        plt.axis('off')
        if i == 0:
            plt.title('Original', pad=10)
    
    # Second row: Latent vectors
    imgs = z_img.cpu().detach().numpy()
    for i, item in enumerate(imgs):
        if i >= 9: break
        plt.subplot(3, 9, i+10)  # Position 10-18 for second row
        item = item.reshape(-1, 8, 8)
        plt.imshow(item[0])
        plt.axis('off')
        if i == 0:
            plt.title('Latent', pad=10)

    # Third row: Reconstructed images
    recon = x_reconstructed.cpu().detach().numpy()
    for i, item in enumerate(recon):
        if i >= 9: break
        plt.subplot(3, 9, i+19)  # Position 19-27 for third row
        item = item.reshape(-1, 28, 28)
        plt.imshow(item[0])
        plt.axis('off')
        if i == 0:
            plt.title('Reconstructed', pad=10)

    plt.tight_layout()
    plt.show()