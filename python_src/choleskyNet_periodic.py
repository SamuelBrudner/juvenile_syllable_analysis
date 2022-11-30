import torch
import torch.nn as nn
import torch.nn.functional as F


class CHOLESKY(nn.Module):
    def __init__(self, lr=1e-3, device_name="auto", latent_size=6):
        super(CHOLESKY, self).__init__()
        self.latent_size = latent_size
        self.tril_size = int((latent_size ** 2 + latent_size) / 2)
        self.lr = lr

        self.fc1 = nn.Linear(3, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.mu_layer = nn.Linear(256, latent_size)
        self.tril_layer = nn.Linear(256, self.tril_size)
        self.diag_layer = nn.Linear(256, latent_size)

        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_name)
        self.to(self.device)

    def calculate_fit(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        mu = self.mu_layer(x)
        batch_size = mu.shape[0]
        tril_vector = self.tril_layer(x)
        tril = torch.zeros((batch_size, self.latent_size, self.latent_size)).to(
            "cuda:0"
        )
        tril_indices = torch.tril_indices(
            row=self.latent_size, col=self.latent_size, offset=0
        )  # index pairs

        tril[:, tril_indices[0], tril_indices[1]] = tril_vector
        tril_T = torch.transpose(tril, 1, 2)
        diag = torch.exp(self.diag_layer(x)) + 1e-6
        sigma = torch.matmul(tril, tril_T)
        sigma = sigma + torch.diag_embed(diag)
        return mu, sigma

# Alternate architecture with fewer nodes per layer
class CHOLESKY_64(nn.Module):
    def __init__(self, lr=1e-3, device_name="auto", latent_size=6):
        super(CHOLESKY_64, self).__init__()
        self.latent_size = latent_size
        self.tril_size = int((latent_size ** 2 + latent_size) / 2)
        self.lr = lr

        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(3, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.mu_layer = nn.Linear(64, latent_size)
        self.tril_layer = nn.Linear(64, self.tril_size)
        self.diag_layer = nn.Linear(64, latent_size)

        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_name)
        self.to(self.device)

    def calculate_fit(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        mu = self.mu_layer(x)
        batch_size = mu.shape[0]
        tril_vector = self.tril_layer(x)
        tril = torch.zeros((batch_size, self.latent_size, self.latent_size)).to(
            "cuda:0"
        )
        tril_indices = torch.tril_indices(
            row=self.latent_size, col=self.latent_size, offset=0
        )  # index pairs

        tril[:, tril_indices[0], tril_indices[1]] = tril_vector
        tril_T = torch.transpose(tril, 1, 2)
        diag = torch.exp(self.diag_layer(x)) + 1e-6
        sigma = torch.matmul(tril, tril_T)
        sigma = sigma + torch.diag_embed(diag)
        return mu, sigma

# Alternate architecture with fewer layers
class CHOLESKY_LAYERS(nn.Module):
    def __init__(self, lr=1e-3, device_name="auto", latent_size=6):
        super(CHOLESKY_LAYERS, self).__init__()
        self.latent_size = latent_size
        self.tril_size = int((latent_size ** 2 + latent_size) / 2)
        self.lr = lr

        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(3, 256)
        self.fc2 = nn.Linear(256, 256)
        self.mu_layer = nn.Linear(256, latent_size)
        self.tril_layer = nn.Linear(256, self.tril_size)
        self.diag_layer = nn.Linear(256, latent_size)

        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_name)
        self.to(self.device)

    def calculate_fit(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.mu_layer(x)
        batch_size = mu.shape[0]
        tril_vector = self.tril_layer(x)
        tril = torch.zeros((batch_size, self.latent_size, self.latent_size)).to(
            "cuda:0"
        )
        tril_indices = torch.tril_indices(
            row=self.latent_size, col=self.latent_size, offset=0
        )  # index pairs

        tril[:, tril_indices[0], tril_indices[1]] = tril_vector
        tril_T = torch.transpose(tril, 1, 2)
        diag = torch.exp(self.diag_layer(x)) + 1e-6
        sigma = torch.matmul(tril, tril_T)
        sigma = sigma + torch.diag_embed(diag)
        return mu, sigma
