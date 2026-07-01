
import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.nn import (
    GATConv, GCNConv, GPSConv, GINConv,
    SAGEConv, LGConv, GATv2Conv, GCN2Conv, GraphConv
)

class SAGEConv_Encoder(torch.nn.Module):

    def __init__(self, in_channels, out_channels):
        super(SAGEConv_Encoder, self).__init__()
        self.conv1 = SAGEConv(in_channels, out_channels, normalize=True)
        self.conv2 = SAGEConv(out_channels, out_channels, normalize=True)

    def forward(self, x, edge_index):

        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        return x


class SAGEConv_Decoder(torch.nn.Module):

    def __init__(self, in_channels, out_channels):
        super(SAGEConv_Decoder, self).__init__()
        self.conv1 = SAGEConv(in_channels, in_channels, normalize=True)
        self.conv2 = SAGEConv(in_channels, out_channels, normalize=True)

    def forward(self, x, edge_index):

        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        return x