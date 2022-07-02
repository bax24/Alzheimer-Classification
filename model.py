import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Union
from monai.networks.nets import ViT

from typing import Sequence, Union

import torch
import torch.nn as nn

from monai.networks.blocks.patchembedding import PatchEmbeddingBlock
from monai.networks.blocks.transformerblock import TransformerBlock

def conv_block_3d(in_dim, out_dim, stride=1, padding=1, batch_norm=True):
    '''
    A standard 3d Conv block
    :param in_dim: in_channels
    :param out_dim:  out_channels
    :param stride:  stride
    :param padding:  padding
    :param batch_norm: whether use bn
    :return: model itself
    '''
    if batch_norm:
        conv_block = nn.Sequential(
            nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=stride, padding=padding),
            # nn.ReLU(),
            nn.BatchNorm3d(out_dim),
            nn.ReLU()
        )
    else:
        conv_block = nn.Sequential(
            nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=stride, padding=padding),
            nn.ReLU()
        )
    return conv_block


def baseline_conv_layers(init_kernel=16):
    '''
    The network baseline
    :param init_kernel:
    :return: model itself
    '''
    bl_conv = nn.Sequential(
        # Conv1
        conv_block_3d(1, init_kernel),
        nn.MaxPool3d(2, stride=2),
    )
    return bl_conv


# for network A, as A have 2 input channels
def baseline_conv_layers_A(init_kernel=16):
    '''
    The network baseline
    :param init_kernel:
    :return: model itself
    '''
    bl_conv = nn.Sequential(
        # Conv1
        conv_block_3d(1, init_kernel),
        nn.MaxPool3d(2, stride=2),
        # Conv2
        conv_block_3d(init_kernel, 2 * init_kernel),
        nn.MaxPool3d(2, stride=2),
        # Conv3
        conv_block_3d(2 * init_kernel, 4 * init_kernel),
        conv_block_3d(4 * init_kernel, 4 * init_kernel),
        nn.MaxPool3d(2, stride=2),
        # Conv4
        conv_block_3d(4 * init_kernel, 8 * init_kernel),
        conv_block_3d(8 * init_kernel, 8 * init_kernel),
        nn.MaxPool3d(2, stride=2),
        # Conv5
        conv_block_3d(8 * init_kernel, 8 * init_kernel),
        conv_block_3d(8 * init_kernel, 8 * init_kernel),
        nn.MaxPool3d(2, stride=2)
    )
    return bl_conv


class NetworkRoi(nn.Module):
    def __init__(self, init_kernel, device, n_output=2):
        super(NetworkRoi, self).__init__()
        self.init_kernel = init_kernel
        self.device = device
        self.n_output = n_output

        # share conv kernels
        self.conv = baseline_conv_layers_A(init_kernel)

        # fc layers
        # 3 * 3 * 1 * 8 * kernel = 2304  --> 512
        # kernel * 384 = 12288
        self.fc = nn.Sequential(
            # nn.Linear(384 * init_kernel, 2304),
            # nn.ReLU(),
            nn.Linear(3 * 3 * 1 * 8 * init_kernel, 512),
            # nn.ReLU(),
            nn.Dropout(),
            # nn.Linear(2304, 512),
            nn.Linear(512, 10),
            # nn.ReLU(),
            nn.Dropout(),
            nn.Linear(10, n_output),
            # nn.Sigmoid()
        )

    def forward(self, inputs):
        mri, label = inputs

        # [B, 48, 96, 96] -> [B, 1, 48, 96, 96]
        mri = mri.unsqueeze(1)
        # pet = pet.unsqueeze(1)

        # img = torch.cat([mri, pet], 1)

        img_feat = self.conv(mri)

        img_feat = img_feat.view(mri.size(0), -1)
        fc_out = self.fc(img_feat)
        # softmax
        output = F.log_softmax(fc_out)
        # output = output.squeeze()
        return output


class NetworkFull(nn.Module):
    def __init__(self, init_kernel, device, n_output=2):
        super(NetworkFull, self).__init__()
        self.init_kernel = init_kernel
        self.device = device
        self.n_output = n_output

        # share conv kernels
        self.conv = baseline_conv_layers_A(init_kernel)

        # fc layers
        # 3 * 3 * 1 * 8 * kernel = 2304  --> 512
        # kernel * 384 = 12288
        self.fc = nn.Sequential(
            nn.Linear(384 * init_kernel, 2304),
            # nn.ReLU(),
            nn.Linear(2304, 512),
            # nn.ReLU(),
            nn.Dropout(),
            # nn.Linear(2304, 512),
            nn.Linear(512, 10),
            # nn.ReLU(),
            nn.Dropout(),
            nn.Linear(10, n_output),
            # nn.Sigmoid()
        )

    def forward(self, inputs):
        mri, label = inputs

        # [B, 48, 96, 96] -> [B, 1, 48, 96, 96]
        mri = mri.unsqueeze(1)
        # pet = pet.unsqueeze(1)

        # img = torch.cat([mri, pet], 1)

        img_feat = self.conv(mri)

        img_feat = img_feat.view(mri.size(0), -1)
        fc_out = self.fc(img_feat)
        # softmax
        output = F.log_softmax(fc_out)
        # output = output.squeeze()
        return output


class NetworkB(nn.Module):

    def __init__(self,
                 in_channel: int,
                 out_channel: int,
                 img_size: Tuple[int, int, int],
                 feature_size: int = 16,
                 hidden_size: int = 768,
                 mlp_dim: int = 3072,
                 num_heads: int = 12,
                 pos_embed: str = "perceptron",
                 norm_name: Union[Tuple, str] = "instance",
                 conv_block: bool = False,
                 res_block: bool = True,
                 num_classes: int = 2,
                 dropout_rate: float = 0.0,
                 ):
        super().__init__()

        self.patch_size = (16, 16, 16)
        self.num_layers = 12
        self.classification = True
        self.vit = ViT(
            in_channels=in_channel,
            img_size=img_size,
            patch_size=self.patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=self.num_layers,
            num_heads=num_heads,
            pos_embed=pos_embed,
            classification=self.classification,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
        )
        self.classification_head = nn.Sequential(nn.Linear(768, 2))
        #self.layer_out = nn.Softmax()

    def forward(self, input):
        mri, label = input
        mri = mri.unsqueeze(1)
        x, hidden_states_out = self.vit(mri)
        output = F.log_softmax(x)
        #output = self.layer_out(x)

        return output

class ViT(nn.Module):
    """
    Vision Transformer (ViT), based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"

    ViT supports Torchscript but only works for Pytorch after 1.8.
    """
    def __init__(
        self,
        in_channels: int,
        img_size: Union[Sequence[int], int],
        patch_size: Union[Sequence[int], int],
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_layers: int = 12,
        num_heads: int = 12,
        pos_embed: str = "conv",
        classification: bool = False,
        num_classes: int = 2,
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            img_size: dimension of input image.
            patch_size: dimension of patch size.
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer.
            num_layers: number of transformer blocks.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            classification: bool argument to determine if classification is used.
            num_classes: number of classes if classification is used.
            dropout_rate: faction of the input units to drop.
            spatial_dims: number of spatial dimensions.

        Examples::

            # for single channel input with image size of (96,96,96), conv position embedding and segmentation backbone
            #>>> net = ViT(in_channels=1, img_size=(96,96,96), pos_embed='conv')

            # for 3-channel with image size of (128,128,128), 24 layers and classification backbone
            #>>> net = ViT(in_channels=3, img_size=(128,128,128), pos_embed='conv', classification=True)

            # for 3-channel with image size of (224,224), 12 layers and classification backbone
            #>>> net = ViT(in_channels=3, img_size=(224,224), pos_embed='conv', classification=True, spatial_dims=2)

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.classification = classification
        self.patch_embedding = PatchEmbeddingBlock(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            pos_embed=pos_embed,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
        )
        self.blocks = nn.ModuleList(
            [TransformerBlock(hidden_size, mlp_dim, num_heads, dropout_rate) for i in range(num_layers)]
        )
        self.norm = nn.LayerNorm(hidden_size)
        if self.classification:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
            self.classification_head = nn.Sequential(nn.Linear(hidden_size, num_classes), nn.Tanh())


    def forward(self, x):
        x = self.patch_embedding(x)
        if hasattr(self, "cls_token"):
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
        hidden_states_out = []
        for blk in self.blocks:
            x = blk(x)
            hidden_states_out.append(x)
        x = self.norm(x)
        if hasattr(self, "classification_head"):
            x = self.classification_head(x[:, 0])
        return x, hidden_states_out
