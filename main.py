import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from model import NetworkB
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch import optim
import argparse
import torch.nn as nn
from model import NetworkA
import trainModelCNN as f
import trainModel as t
from collections import Counter
import time


def get_data(train_size, disp=False):
    # Training dataset
    MRI_train = []
    labels_train = []

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(torch.Tensor(mean))
    ])

    # Testing dataset
    MRI_test = []
    labels_test = []

    diagnosis = ['AD', 'CN']
    for diag in diagnosis:
        index = 0
        for path, dirs, files in os.walk(os.path.join(args.data, diag)):
            # Divide both classes evenly between train and test dataset
            upper_bound_train = int(len(files) * train_size)
            for filename in files:
                if filename == '.DS_Store':
                    continue
                image = os.path.join(args.data, diag, filename)
                sitk_image = sitk.ReadImage(image)

                # transform into a numpy array
                MRI = sitk.GetArrayFromImage(sitk_image)
                # add the Channel dimension
                # MRI = MRI[np.newaxis, ...]
                # Put sitk in right dataset
                if index < upper_bound_train:
                    # Train section
                    MRI = transform(MRI)
                    MRI_train.append(MRI)
                    labels_train.append(diag)
                    index += 1
                else:
                    MRI_test.append(MRI)
                    labels_test.append(diag)

    if disp:
        print("Training : ", Counter(labels_train))
        print("Testing : ", Counter(labels_test))

    labels_train = torch.tensor([1 if x == 'AD' else 0 for x in labels_train])
    labels_test = torch.tensor([1 if x == 'AD' else 0 for x in labels_test])
    train_db = list(zip(MRI_train, labels_train))
    test_db = list(zip(MRI_test, labels_test))

    return train_db, test_db


class MRIDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        self.__MRI = []
        self.__label = []

        self.neg = 'CN'
        self.pos = 'AD'

        self.root_dir = root_dir
        self.transform = transform

        label_dict = {self.neg: 0, self.pos: 1}

        for diag in ['AD', 'CN']:
            path = os.path.join(root_dir, diag)
            for _, _, files in os.walk(path):
                for filename in files:
                    if filename == ".DS_Store":
                        continue
                    self.__MRI.append(os.path.join(root_dir, diag, filename))
                    self.__label.append(label_dict[diag])

    def __getitem__(self, index):
        mri_image = sitk.ReadImage(self.__MRI[index])
        mri_image = sitk.GetArrayFromImage(mri_image)
        label = self.__label[index]

        return mri_image, label

    def __len__(self):
        return len(self.__MRI)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='AD Classifier')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--batches', type=int, default=10, metavar='N',
                        help='input batch size for training (default: 10)')
    parser.add_argument('--data', type=str, default='S_C/', metavar='str',
                        help='folder that contains data (default: S_C)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='the learning rate (default: 0.001')
    parser.add_argument('--network', type=str, default='CNN', metavar='str',
                        help='the network name (default: CNN)')
    parser.add_argument('--optim', type=str, default='ADAM', metavar='str',
                        help='the optimizer')

    args = parser.parse_args()
    print("Args: ", args)

    id = args.network + str(args.epochs) + str('-') + str(args.batches) + str('-') + args.optim[0]

    # GPU and CUDA
    print("The number of GPUs:", torch.cuda.device_count())

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")

    # ----------------
    # Define my model
    # ----------------
    # 3D CNN
    if args.network == 'CNN':
        model = NetworkA(init_kernel=32, device=device)
    else:
        model = NetworkB(in_channel=1, out_channel=1, img_size=(128, 128, 96), pos_embed='conv')

    model.to(device)

    if torch.cuda.device_count() > 0:
        print("Using MultiGPUs")
        model = nn.DataParallel(model)

    # Define my optimizer
    if args.optim == 'ADAM':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0005)
    else:
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr)

    # Define my Loss
    loss = nn.BCELoss()  # Not used for ViT

    # Downloading datasets

    dataset = MRIDataset(root_dir=args.data, transform=transforms.ToTensor())

    train_size = int(0.7 * len(dataset))
    test_size = len(dataset) - train_size
    train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])

    #train_set, test_set = get_data(train_size=0.7, disp=True)

    train_loader = DataLoader(train_set, batch_size=args.batches, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batches, shuffle=True)

    train_loss = []
    test_loss = []

    for epoch in range(1, args.epochs):
        print("--------------------")
        print("EPOCH " + str(epoch))
        print("--------------------")
        total_loss = f.train(model, train_loader, optimizer, epoch, device)
        train_loss.append(total_loss)
        print("\nTraining ok ! With a loss of ", total_loss)
        total_loss = f.test(model, test_loader, epoch, device)
        test_loss.append(total_loss)
        print("Testing ok ! With a loss of \n", total_loss)

    # Plotting Testing loss
    name = str('E') + str('_') + id
    plt.Figure(figsize=(13, 5))
    plt.title('Evolution of Loss curves')
    ax = plt.gca()
    ax.set_ylim([0, 30])
    plt.plot(train_loss, label='Training')
    plt.plot(test_loss, label='Testing')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc="upper right")
    plt.savefig('Figures/' + name)
