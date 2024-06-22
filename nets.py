import torch
import torch.nn as nn
import torch.nn.functional as F
class RepresentationNetwork(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        def size_linear_unit(size, kernel_size=3, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1
        num_linear_units = size_linear_unit(10) * size_linear_unit(10) * 16
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=num_linear_units, out_features=out_channels),
            nn.ReLU(),
            nn.Linear(in_features=out_channels, out_features=out_channels)
        )
    
    def forward(self, x):
        return self.net(x)

class DynamicsNetwork(nn.Module):

    def __init__(self, num_channels):
        super().__init__()
        self.embedding_net = nn.Sequential(
            nn.Linear(in_features=num_channels+1, out_features=num_channels),
            nn.ReLU(),
            nn.Linear(in_features=num_channels, out_features=num_channels),
            nn.ReLU(),
            nn.Linear(in_features=num_channels, out_features=num_channels)
        )
        self.reward_net = nn.Sequential(
            nn.Linear(in_features=num_channels, out_features=num_channels),
            nn.ReLU(),
            nn.Linear(in_features=num_channels, out_features=1),
            nn.Flatten(0)
        )
        self.done_net = nn.Sequential(
            nn.Linear(in_features=num_channels, out_features=num_channels),
            nn.ReLU(),
            nn.Linear(in_features=num_channels, out_features=1),
            nn.Flatten(0)
        )
        self.dynamics_net = nn.Sequential(
            nn.Linear(in_features=num_channels+1, out_features=num_channels),
            nn.ReLU(),
            nn.Linear(in_features=num_channels, out_features=num_channels),
            nn.ReLU(),
            nn.Linear(in_features=num_channels, out_features=num_channels)
        )

    
    # for inference
    def forward(self, x):
        predicted_next_latent_state = self.dynamics_net(x)
        embedding = self.embedding_net(x)
        reward = self.reward_net(embedding)
        done = self.done_net(embedding)
        return predicted_next_latent_state, reward, nn.Sigmoid()(done)


class DynamicsNetworkDropout(nn.Module):

    def __init__(self, num_channels, dropout_p=0.05):
        super().__init__()
        self.embedding_net = nn.Sequential(
            nn.Linear(in_features=num_channels + 1, out_features=num_channels),
            nn.ReLU(),
            nn.Linear(in_features=num_channels, out_features=num_channels),
            nn.ReLU(),
            nn.Linear(in_features=num_channels, out_features=num_channels)
        )
        self.reward_net = nn.Sequential(
            nn.Linear(in_features=num_channels, out_features=num_channels),
            nn.Dropout(p=dropout_p),  # Added dropout layer
            nn.ReLU(),
            nn.Linear(in_features=num_channels, out_features=1),
            nn.Flatten(0)
        )
        self.done_net = nn.Sequential(
            nn.Linear(in_features=num_channels, out_features=num_channels),
            nn.ReLU(),
            nn.Linear(in_features=num_channels, out_features=1),
            nn.Flatten(0)
        )
        self.dynamics_net = nn.Sequential(
            nn.Linear(in_features=num_channels + 1, out_features=num_channels),
            nn.ReLU(),
            # nn.Dropout(p=dropout_p),  # Added dropout layer
            nn.Linear(in_features=num_channels, out_features=num_channels),
            nn.ReLU(),
            # nn.Dropout(p=dropout_p),  # Added dropout layer
            nn.Linear(in_features=num_channels, out_features=num_channels)
        )
        self.dropout_p = dropout_p

    # for inference
    def forward(self, x):
        predicted_next_latent_state = self.dynamics_net(x)
        embedding = self.embedding_net(x)
        reward = self.reward_net(embedding)
        done = self.done_net(embedding)
        return predicted_next_latent_state, reward, nn.Sigmoid()(done)

    def dynamics_forward(self, x):
        x = F.relu(
            nn.functional.dropout(F.linear(x, self.dynamics_net[0].weight, self.dynamics_net[0].bias), p=self.dropout_p,
                                  training=True))
        x = F.relu(
            nn.functional.dropout(F.linear(x, self.dynamics_net[2].weight, self.dynamics_net[2].bias), p=self.dropout_p,
                                  training=True))
        x = F.linear(x, self.dynamics_net[4].weight, self.dynamics_net[4].bias)
        return x


# class ReconstructionNetwork(nn.Module):

#     def __init__(self, num_channels):
#         super().__init__()
#         # from a 128-dim vector to a 4x10x10 tensor using nn.ConvTranspose2d after two linear layers
#         self.net = nn.Sequential(
#             nn.Linear(in_features=num_channels, out_features=num_channels),
#             nn.ReLU(),
#             nn.Linear(in_features=num_channels, out_features=num_channels),
#             nn.ReLU(),
#             nn.Linear(in_features=num_channels, out_features=num_channels),
#             nn.ReLU(),
#             nn.Linear(in_features=num_channels, out_features=num_channels),
#             nn.ReLU(),
#             nn.Linear(in_features=num_channels, out_features=num_channels),
#             nn.ReLU(),
#             nn.Linear(in_features=num_channels, out_features=4*10*10),
#             nn.Unflatten(1, (4, 10, 10)),
#             # nn.Sigmoid()
#         )
    
#     def forward(self, x):
#         return self.net(x)
    
class ReconstructionNetwork(nn.Module):

    def __init__(self, in_channels=128, num_channels=64):
        super().__init__()
        self.deconv_layers = nn.Sequential(
            nn.Unflatten(1, (in_channels, 1, 1)),
            # First deconvolution: 128x1x1 to 32x2x2
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=num_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # Second deconvolution: 32x2x2 to 32x5x5
            nn.ConvTranspose2d(in_channels=num_channels, out_channels=num_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # Third deconvolution: 32x5x5 to 4x10x10
            nn.ConvTranspose2d(in_channels=num_channels, out_channels=4, kernel_size=6, stride=2, padding=1),
        )

    def forward(self, x):
        return self.deconv_layers(x)