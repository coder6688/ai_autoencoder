import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
from torchvision import  transforms
import matplotlib.pyplot as plt
from model import LinearAutoencoder
import numpy as np
from util import device

INPUT_DIM = 784
HID1_DIM = 128
HID2_DIM = 64
HID3_DIM = 16
HID4_DIM = 3
LEARNING_RATE = 1e-4 #KARPATHY CONSTANT
BATCH_SIZE = 64
NUM_EPOCHS = 10

OUTPUT_DIR = 'output/lae'

# Define the transformation to convert images to tensors    
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load the model
model = LinearAutoencoder(input_dim=INPUT_DIM, hid1_dim=HID1_DIM, hid2_dim=HID2_DIM, hid3_dim=HID3_DIM, hid4_dim=HID4_DIM).to(device)
model.load_state_dict(torch.load(f'{OUTPUT_DIR}/lae.pth'))
model.eval()    

# Load the latent vector
epoch = 50
z_img = np.load(f'{OUTPUT_DIR}/latent_epoch_{epoch}.npy')
z_img = torch.from_numpy(z_img).float().view(-1, 3).to(device)

x_img = np.load(f'{OUTPUT_DIR}/original_epoch_{epoch}.npy')
x_img = torch.from_numpy(x_img).float().view(-1, 28, 28).to(device)


# Generate the image    
with torch.no_grad():
    x_reconstructed = model.decode(z_img)

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
        item = item.reshape(-1, 1, 3)
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