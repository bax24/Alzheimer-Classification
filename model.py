import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Union
from monai.networks.nets import ViT


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


class NetworkA(nn.Module):
    def __init__(self, init_kernel, device, n_output=2):
        super(NetworkA, self).__init__()
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
            nn.Dropout(),
            nn.Linear(2304, 512),
            nn.Linear(512, 10),
            nn.Dropout(),
            nn.Linear(10, n_output),
        )

    def forward(self, inputs):
        mri, label = inputs

        # [B, 48, 96, 96] -> [B, 1, 48, 96, 96]
        mri = mri.unsqueeze(1)
        #pet = pet.unsqueeze(1)

        #img = torch.cat([mri, pet], 1)

        img_feat = self.conv(mri)

        img_feat = img_feat.view(mri.size(0), -1)
        fc_out = self.fc(img_feat)
        # 在tf的版本里是softmax
        output = F.log_softmax(fc_out)
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
        self.layer_out = nn.Sigmoid()

    def forward(self, input):
        x, hidden_states_out = self.vit(input)
        # output = F.log_softmax(x, dim=1)
        output = self.layer_out(x)

        return output
