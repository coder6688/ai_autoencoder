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
from model import VAE
from util import device

OUTPUT_DIR = 'output/vae'
os.makedirs(OUTPUT_DIR, exist_ok=True)

torch.manual_seed(42)

# configuration
INPUT_DIM = 784
HID_DIM = 200
Z_DIM = 20
LEARNING_RATE = 1e-4 #KARPATHY CONSTANT
BATCH_SIZE = 64
NUM_EPOCHS = 10

# dataset loading
dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
model = VAE(input_dim=INPUT_DIM, hid_dim=HID_DIM, z_dim=Z_DIM).to(device)

# loss function and optimizer
criterion = nn.BCELoss(reduction='sum')
#criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# training loop
for epoch in range(NUM_EPOCHS):
    for batch_data, _ in tqdm(train_loader):

        x = batch_data.view(-1, 28*28).to(device)
        x_reconstructed, mu, std = model(x)

        # reconstruction loss
        recons_loss = criterion(x_reconstructed, x)

        # KL divergence loss
        var = std**2
        kl_loss = -0.5 * torch.sum(1 + torch.log(var) - mu**2 - var)

        # total loss
        loss = recons_loss + kl_loss

        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {loss.item():.4f}')

# save the model
criterion_name = 'BCELoss' if isinstance(criterion, nn.BCELoss) else 'MSELoss'
torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f'vae_{criterion_name}_epoch_{epoch+1}.pth'))
