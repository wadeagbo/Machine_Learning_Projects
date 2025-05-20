# 🧩 1. 📥 Load and Visualize Hyperspectral Data
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np

# Load hyperspectral image and ground truth labels
dataset = loadmat('data/Indian_pines_corrected.mat')['indian_pines_corrected']
ground_truth = loadmat('data/Indian_pines_gt.mat')['indian_pines_gt']

print("Image shape:", dataset.shape)        # (145, 145, 200)
print("Ground truth shape:", ground_truth.shape)  # (145, 145)



# Visualize Band 30
plt.figure(figsize=(8, 6))
plt.imshow(dataset[:, :, 30], cmap='gray')
plt.title("Band 30")
plt.colorbar()
plt.show()

# False-color RGB image using bands 30 (R), 20 (G), 10 (B)
plt.figure(figsize=(8, 6))
r, g, b = 30, 20, 10
rgb_img = dataset[:, :, [r, g, b]]
rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min())  # normalize
plt.imshow(rgb_img)
plt.title("False Color Image")
plt.colorbar()

plt.show()



def plot_band(dataset):
    plt.figure(figsize=(8, 6))
    band_no = np.random.randint(dataset.shape[2])
    plt.imshow(dataset[:,:, band_no], cmap='jet')
    plt.title(f'Band-{band_no}', fontsize=14)
    plt.axis('off')
    plt.colorbar()
    plt.show()

plt.figure(figsize=(8, 6))
plt.imshow(ground_truth, cmap='jet')
plt.axis('off')
plt.colorbar(ticks= range(0,16))
plt.show()

# 🧩 2. 🧪 Flatten Data for Analysis (PCA, ML, DL)

# Flatten spatial dimensions
rows, cols, bands = dataset.shape
X = dataset
X_2d = X.reshape(-1, bands)  # shape: (145*145, 200)
y_1d = ground_truth.ravel()  # shape: (145*145,)

# 🧩 3. 🎯 Filter for Labeled Data Only

mask = y_1d > 0
X_filtered = X_2d[mask]
y_filtered = y_1d[mask]

print("Filtered data shape:", X_filtered.shape)
print("Filtered labels shape:", y_filtered.shape)

# 🧩 4. ⚙️ PCA-Based Denoising
from sklearn.decomposition import PCA

# Apply PCA and reconstruct denoised image
pca = PCA(n_components=20)
X_pca = pca.fit_transform(X_filtered)
X_denoised = pca.inverse_transform(X_pca)

# Optional: visualize one band after PCA
img_pca_reshaped = X_denoised.reshape(-1, bands)  # only if needed

# 🧩 5. 🔧 Denoising Autoencoder (PyTorch)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class DenoisingAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super(DenoisingAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 64))
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, input_dim))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Simulate training data
x_clean = torch.rand(1000, 200)
noise = 0.1 * torch.randn_like(x_clean)
x_noisy = x_clean + noise

trainset = TensorDataset(x_noisy, x_clean)
loader = DataLoader(trainset, batch_size=64, shuffle=True)

# Train model
model = DenoisingAutoencoder(input_dim=200)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(20):
    for noisy, clean in loader:
        output = model(noisy)
        loss = criterion(output, clean)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")



# 🧩 6. 🧼 Total Variation Denoising (Single Band)

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from skimage.restoration import denoise_tv_chambolle

# Load the dataset
data = loadmat('data/Indian_pines_corrected.mat')
dataset = data['indian_pines_corrected']

# Choose a spectral band to denoise
band = 30
noisy_band = dataset[:, :, band] + 0.05 * np.random.randn(*dataset[:, :, band].shape)

# Apply Total Variation denoising
denoised_band = denoise_tv_chambolle(noisy_band, weight=0.1)

# Plot the results
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(noisy_band, cmap='gray')
plt.title("Noisy Band")
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(denoised_band, cmap='gray')
plt.title("TV Denoised Band")
plt.colorbar()

plt.tight_layout()
plt.show()
