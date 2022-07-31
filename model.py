import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Union
# from monai.networks.nets import ViT

from typing import Sequence, Union

import torch
import torch.nn as nn

from monai.networks.blocks.patchembedding import PatchEmbeddingBlock
from monai.networks.blocks.transformerblock import TransformerBlock
from monai.networks.blocks import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock
from monai.networks.blocks.dynunet_block import UnetOutBlock
# from monai.networks.nets import ViT


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
    def __init__(self, init_kernel, device, n_output=1):
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
            nn.ReLU(),
            nn.Dropout(),
            # nn.Linear(2304, 512),
            nn.Linear(512, 10),
            nn.ReLU(),
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
        # output = F.log_softmax(fc_out)
        output = fc_out.squeeze(1)
        return output


class NetworkFull(nn.Module):
    def __init__(self, init_kernel, device, n_output=1):
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
            nn.ReLU(),
            nn.Linear(2304, 512),
            nn.ReLU(),
            nn.Dropout(),
            # nn.Linear(2304, 512),
            nn.Linear(512, 10),
            nn.ReLU(),
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
        # output = F.log_softmax(fc_out)
        output = fc_out.squeeze(1)
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
                 pos_embed: str = "conv",
                 norm_name: Union[Tuple, str] = "instance",
                 conv_block: bool = False,
                 res_block: bool = True,
                 num_classes: int = 1,
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
        # self.classification_head = nn.Sequential(nn.Linear(768, 1))
        # self.layer_out = nn.Softmax()

    def forward(self, input):
        mri, label = input
        mri = mri.unsqueeze(1)
        x, hidden_states_out = self.vit(mri)
        # output = F.log_softmax(x)
        # output = self.layer_out(x)

        return x.squeeze(1)


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
            self.classification_head = nn.Sequential(nn.Linear(hidden_size, num_classes))

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

class UNETR(nn.Module):
    """
    UNETR based on: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_size: Tuple[int, int, int],
        feature_size: int = 16,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_heads: int = 12,
        pos_embed: str = "perceptron",
        norm_name: Union[Tuple, str] = "instance",
        conv_block: bool = False,
        res_block: bool = True,
        dropout_rate: float = 0.0,
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            img_size: dimension of input image.
            feature_size: dimension of network feature size.
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            norm_name: feature normalization type and arguments.
            conv_block: bool argument to determine if convolutional block is used.
            res_block: bool argument to determine if residual block is used.
            dropout_rate: faction of the input units to drop.
        Examples::
            # for single channel input 4-channel output with patch size of (96,96,96), feature size of 32 and batch norm
            >>> net = UNETR(in_channels=1, out_channels=4, img_size=(96,96,96), feature_size=32, norm_name='batch')
            # for 4-channel input 3-channel output with patch size of (128,128,128), conv position embedding and instance norm
            >>> net = UNETR(in_channels=4, out_channels=3, img_size=(128,128,128), pos_embed='conv', norm_name='instance')
        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise AssertionError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise AssertionError("hidden size should be divisible by num_heads.")

        if pos_embed not in ["conv", "perceptron"]:
            raise KeyError(f"Position embedding layer of type {pos_embed} is not supported.")

        self.num_layers = 12
        self.patch_size = (16, 16, 16)
        self.feat_size = (
            img_size[0] // self.patch_size[0],
            img_size[1] // self.patch_size[1],
            img_size[2] // self.patch_size[2],
        )
        self.hidden_size = hidden_size
        self.classification = False
        self.vit = ViT(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=self.patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=self.num_layers,
            num_heads=num_heads,
            pos_embed=pos_embed,
            classification=self.classification,
            dropout_rate=dropout_rate,
        )
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrPrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=feature_size * 2,
            num_layer=2,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder3 = UnetrPrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=feature_size * 4,
            num_layer=1,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder4 = UnetrPrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.decoder5 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        #self.out = UnetOutBlock(spatial_dims=3, in_channels=feature_size, out_channels=out_channels)  # type: ignore
        self.out = nn.Linear(8, 1)

    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def load_from(self, weights):
        with torch.no_grad():
            res_weight = weights
            # copy weights from patch embedding
            for i in weights["state_dict"]:
                print(i)
            self.vit.patch_embedding.position_embeddings.copy_(
                weights["state_dict"]["module.transformer.patch_embedding.position_embeddings_3d"]
            )
            self.vit.patch_embedding.cls_token.copy_(
                weights["state_dict"]["module.transformer.patch_embedding.cls_token"]
            )
            self.vit.patch_embedding.patch_embeddings[1].weight.copy_(
                weights["state_dict"]["module.transformer.patch_embedding.patch_embeddings.1.weight"]
            )
            self.vit.patch_embedding.patch_embeddings[1].bias.copy_(
                weights["state_dict"]["module.transformer.patch_embedding.patch_embeddings.1.bias"]
            )

            # copy weights from  encoding blocks (default: num of blocks: 12)
            for bname, block in self.vit.blocks.named_children():
                print(block)
                block.loadFrom(weights, n_block=bname)
            # last norm layer of transformer
            self.vit.norm.weight.copy_(weights["state_dict"]["module.transformer.norm.weight"])
            self.vit.norm.bias.copy_(weights["state_dict"]["module.transformer.norm.bias"])

    def forward(self, x_in):
        x, hidden_states_out = self.vit(x_in)
        enc1 = self.encoder1(x_in)
        x2 = hidden_states_out[3]
        enc2 = self.encoder2(self.proj_feat(x2, self.hidden_size, self.feat_size))
        x3 = hidden_states_out[6]
        enc3 = self.encoder3(self.proj_feat(x3, self.hidden_size, self.feat_size))
        x4 = hidden_states_out[9]
        enc4 = self.encoder4(self.proj_feat(x4, self.hidden_size, self.feat_size))
        logits = self.out(enc4)
        return logits

