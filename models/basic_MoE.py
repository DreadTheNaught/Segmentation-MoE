import torch.nn as nn
import torch


class Expert(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(Expert, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 1, 3, padding=1), nn.Sigmoid()
        )

    def forward(self, x):
        e1 = self.encoder(x)
        d1 = self.decoder(e1)
        return e1, d1


class Expertlayer(nn.Module):
  def __init__(self, expert_num, in_channels, out_channels):
    super(Expertlayer, self).__init__()
    self.expert_num = expert_num
    self.experts = nn.ModuleList(
        [Expert(in_channels, out_channels) for _ in range(self.expert_num)])

  def forward(self, x_list):
    assert len(x_list) == self.expert_num
    e_list = []
    d_list = []
    for loop in range(self.expert_num):
      encoder, decoder = self.experts[loop](x_list[loop])
      e_list.append(encoder)
      d_list.append(decoder)

    return e_list, d_list


class DecoderLayer(nn.Module):
  def __init__(self, out_channels):
    super(DecoderLayer, self).__init__()
    self.out_channels = out_channels
    self.decoder = nn.Sequential(
        nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
        nn.Conv2d(64, self.out_channels, 3, padding=1), nn.ReLU()
    )

  def forward(self, x):
      return self.decoder(x)


class Router(nn.Module):
  def __init__(self, input_expert_num, input_out_channels, output_expert_num):
    super(Router, self).__init__()
    self.input_expert_num = input_expert_num
    self.decoder = nn.ModuleList(
        [DecoderLayer(input_out_channels) for _ in range(output_expert_num)])

    self.router_layer = nn.Sequential(
        nn.Conv2d(input_out_channels, 64, 3, padding=1), nn.ReLU(),
        nn.Flatten(),
        # This part is still hardcoded based on the image size
        nn.Linear(64 * 224 * 224, output_expert_num),
        nn.Sigmoid()
    )

  def forward(self, x_list):
    decoder_outputs = []
    for loop in range(len(self.decoder)):
      decoder_outputs.append(self.decoder[loop](x_list[loop]))

    router_input = torch.stack(decoder_outputs).sum(dim=0)
    router_output = self.router_layer(router_input)
    # Added batch dimension to weights
    weights = router_output.view(
        router_output.shape[0], self.input_expert_num, 1, 1)
    stacked_outputs = torch.stack(decoder_outputs)
    weighted_outputs = weights * stacked_outputs
    summed_outputs = weighted_outputs.sum(dim=0)
    return summed_outputs


class MoE(nn.Module):
  def __init__(self, input_expert_num, input_out_channels, output_expert_num):
    super(MoE, self).__init__()
    self.input_expert_num = input_expert_num
    self.expert_layer1 = Expertlayer(
        input_expert_num, input_out_channels, input_out_channels)
    self.router1 = Router(
        input_expert_num, input_out_channels, output_expert_num)
    self.expert_layer2 = Expertlayer(
        input_expert_num, input_out_channels, input_out_channels)

  def forward(self, x):
    x_list = [x for i in range(self.input_expert_num)]
    e1, d1 = self.expert_layer1(x_list)
    r1 = self.router1(e1)
    r1_list = [r1 for i in range(self.input_expert_num)]
    e2, d2 = self.expert_layer2(r1_list)
    decoder_outputs = [d1, d2]
    d1, d2 = torch.stack(d1).sum(dim=0), torch.stack(
        d2).sum(dim=0)  # This part is changed
    all_seg_maps = torch.cat((d1, d2), dim=1)
    final_segmentation_map = all_seg_maps.sum(dim=1)
    final_segmentation_map = final_segmentation_map.unsqueeze(
        1)  # making sure the dimensions are consistent
    return final_segmentation_map, decoder_outputs
