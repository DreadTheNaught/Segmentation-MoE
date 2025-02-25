import torch.nn as nn
import torch

class Expert(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Expert, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, out_channels, 3, padding=1), nn.Sigmoid()
        )

    def forward(self, x):
        e1 = self.encoder(x)
        d1 = self.decoder(e1)
        return d1


class Router(nn.Module):
    def __init__(self, in_channels):
        super(Router, self).__init__()
        self.router_layer = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1), nn.ReLU(),
            nn.Flatten(),
            # This part is still hardcoded based on the image size
            nn.Linear(64 * 224 * 224, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.router_layer(x)


class Flat_MoE(nn.Module):
    def __init__(self, in_channels, number_of_experts):
        super(Flat_MoE, self).__init__()
        self.in_channels = in_channels
        self.number_of_experts = number_of_experts
        self.expert_list = nn.ModuleList(
            [Expert(self.in_channels, 1) for _ in range(self.number_of_experts)])
        self.router_list = nn.ModuleList(
            [Router(self.in_channels) for _ in range(self.number_of_experts)])

    def forward(self, x):
        partial_segmentation_maps = []
        for expert_number in range(self.number_of_experts):
            router_output = self.router_list[expert_number](x)
            expert_output = self.expert_list[expert_number](x)
            # expert_output = router_output * expert_output
            expert_output = router_output.detach().view(-1, 1, 1, 1) * expert_output
            partial_segmentation_maps.append(expert_output)

        output = sum(partial_segmentation_maps)
        partial_segmentation_maps = torch.stack(
            partial_segmentation_maps, dim=0)
        return output, partial_segmentation_maps
