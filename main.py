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
from model import NetworkRoi, NetworkFull, UNETR
import trainModelCNN as f
from collections import Counter
import time
import monai
from torch.utils.tensorboard import SummaryWriter
import torchvision
from PIL import Image

writer = SummaryWriter("runs/mri")


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


def get_norm(data):
    Mri_train = []

    diagnosis = ['AD', 'CN']
    for diag in diagnosis:
        for path, dirs, files in os.walk(os.path.join(args.data, diag)):
            for filename in files:
                if filename == '.DS_Store':
                    continue
                image = os.path.join(args.data, diag, filename)
                sitk_image = sitk.ReadImage(image)

                # transform into a numpy array
                MRI = sitk.GetArrayFromImage(sitk_image)
                Mri_train.append(MRI)

    mri_arr = np.array(Mri_train)
    mri_tensor = torch.tensor(mri_arr)
    means = mri_tensor.mean(dim=0, keepdim=True)
    means = means.squeeze(0)
    stds = mri_tensor.std(dim=0, keepdim=True)
    stds = stds.squeeze(0)
    stds[stds == 0] = 1

    return np.array(means), np.array(stds)


class MRIDataset(Dataset):

    def __init__(self, means, stds):
        self.__MRI = []
        self.__label = []

        if args.classification == 1:
            diagnosis = ['AD', 'CN']
            self.neg = 'CN'
            self.pos = 'AD'
        if args.classification == 2:
            diagnosis = ['AD', 'MCI']
            self.neg = 'MCI'
            self.pos = 'AD'

        self.root_dir = args.data
        # self.transform = transform

        label_dict = {self.neg: 0, self.pos: 1}

        for diag in diagnosis:
            path = os.path.join(self.root_dir, diag)
            for _, _, files in os.walk(path):
                for filename in files:
                    if filename == ".DS_Store":
                        continue
                    self.__MRI.append(os.path.join(self.root_dir, diag, filename))
                    self.__label.append(label_dict[diag])

    def __getitem__(self, index):
        mri_image = sitk.ReadImage(self.__MRI[index])
        mri_image = sitk.GetArrayFromImage(mri_image)
        # mri_tensor = torch.tensor(mri_image)
        normalized_mri = (mri_image - means) / stds
        label = self.__label[index]

        return mri_image, label

    def __len__(self):
        return len(self.__MRI)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='AD Classifier')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--batches', type=int, default=10, metavar='N',
                        help='input batch size for training (default: 10)')
    parser.add_argument('--data', type=str, default='Batch/', metavar='str',
                        help='folder that contains data (default: S_C)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='the learning rate (default: 0.01')
    parser.add_argument('--kernel', type=int, default=16, metavar='N',
                        help='the kernel size (default: 16')
    parser.add_argument('--network', type=str, default='CNN_Full', metavar='str',
                        help='the network name (default: CNN)')
    parser.add_argument('--optim', type=str, default='ADAM', metavar='str',
                        help='the optimizer (default: Adam)')
    parser.add_argument('--classification', type=str, default=1, metavar='N',
                        help='What would you like to classify ? (default: AD vs CN)')
    parser.add_argument('--tag', type=str, default='', metavar='str',
                        help='a tag')

    args = parser.parse_args()
    print("Args: ", args)

    id = args.network + str(args.epochs) + str('-') + str(args.batches) + str('-') + args.optim[0] + str('-') \
         + args.data + str('-') + str(args.kernel) + str('-') + args.tag

    # GPU and CUDA
    print("The number of GPUs:", torch.cuda.device_count())

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")

    # ----------------
    # Define my model
    # ----------------
    # 3D CNN
    if args.network == 'CNN_Roi':
        model = NetworkRoi(init_kernel=args.kernel, device=device)
    elif args.network == 'CNN_Full':
        model = NetworkFull(init_kernel=args.kernel, device=device)
    elif args.network == 'ViT':
        model = NetworkB(in_channel=1, out_channel=1, img_size=(96, 96, 48), pos_embed='conv')
        model.classification_head = nn.Sequential(nn.Linear(768, 1))
    elif args.network == 'Chris':
        model = monai.networks.nets.ViT(in_channels=1, patch_size=(16, 16, 16), img_size=(96, 96, 48), pos_embed='conv',
                                        num_classes=2, hidden_size=768, mlp_dim=3072, classification=True, num_heads=12,
                                        dropout_rate=0.0)
        model.classification_head = nn.Sequential(nn.Linear(768, 1))
    elif args.network == 'test':
        model = monai.networks.nets.ViT(in_channels=1, img_size=(128, 128, 96), pos_embed='conv', classification=True,
                                        patch_size=(16, 16, 16))
    elif args.network == 'unetr':
        model = UNETR(in_channels=4, out_channels=3, img_size=(128, 128, 96), pos_embed='conv', norm_name='instance')

    model.to(device)

    if torch.cuda.device_count() > 0:
        print("Using MultiGPUs")
        model = nn.DataParallel(model)

    # Define my optimizer
    if args.optim == 'ADAM':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    else:
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr)

    # Define my Loss
    criterion = nn.BCEWithLogitsLoss()

    # Downloading datasets

    means, stds = get_norm(args.data)

    dataset = MRIDataset(means, stds)

    train_size = int(0.7 * len(dataset))
    test_size = len(dataset) - train_size
    train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])

    print("Training : ", train_size)
    print("Testing : ", test_size)

    train_classes = [label for _, label in train_set]
    print("Train labels distribution :", Counter(train_classes))

    test_classes = [label for _, label in test_set]
    print("Test labels distribution : ", Counter(test_classes))

    # means = train_set.mean(dim=1, keepdim=True)
    # stds = train_set.std(dim=1, keepdim=True)
    # normalized_data = (train_set - means) / stds

    train_loader = DataLoader(train_set, batch_size=args.batches, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=test_size, shuffle=True)

    train_loss = []
    test_loss = []

    step = 0

    for epoch in range(1, args.epochs):
        print("--------------------")
        print("EPOCH " + str(epoch))
        print("--------------------")
        # total_loss = f.train(model, train_loader, optimizer, epoch, criterion, device)
        # Training
        model.train()
        losses = []
        accuracies = []
        correct = 0
        for batch_idx, (mri, label) in enumerate(train_loader):
            # Get data to cuda if possible
            mri, label = mri.to(device), label.to(device)

            # label = label.unsqueeze(1)
            data = (mri.type(torch.FloatTensor), label.type(torch.FloatTensor))

            # Forward
            output = model(data)
            label = label.to(torch.float32)
            loss = criterion(output, label)
            losses.append(loss.item())

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate running training accuracy
            pred = (torch.sigmoid(output) > 0.5).float()
            correct = pred.eq(label.view_as(pred)).sum().item()
            running_train_acc = float(correct) / float(mri.shape[0])

            # if batch_idx % 10 == 0:
            # step = (epoch - 1) * args.batches
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * 10, len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

            print("Correct = ", correct)
            writer.add_scalar('training loss/' + id, loss, global_step=step)
            writer.add_scalar('training accuracy/' + id, running_train_acc, global_step=step)
            step += 1

        # print("\nTraining ok ! With a loss of ", total_loss)
        # writer.add_scalar('training loss/'+id, total_loss, global_step=epoch)
        # writer.add_scalar('training accuracy/' + id, running_train_acc, global_step=step)
        # total_loss = f.test(model, test_loader, epoch, criterion,  device)

        # Testing
        # test_loss.append(total_loss)
        model.eval()
        test_loss = 0
        correct = 0
        TP = 0
        TN = 0
        FP = 0
        FN = 0

        with torch.no_grad():
            for mri, label in test_loader:
                mri, label = mri.to(device), label.to(device)
                # label = label.unsqueeze(1)
                data = (mri.type(torch.FloatTensor), label.type(torch.FloatTensor))
                output = model(data)

                label = label.to(torch.float32)
                test_loss += criterion(output, label).item()
                pred = (torch.sigmoid(output) > 0.5).float()
                correct += pred.eq(label.view_as(pred)).sum().item()
                truths = pred.eq(label.view_as(pred))

                for i, right in enumerate(truths):
                    if right:
                        if label[i]:
                            TP += 1
                        else:
                            TN += 1
                    else:
                        if label[i]:
                            FN += 1
                        else:
                            FP += 1

        lenght = len(test_loader.dataset)
        print("length = ", lenght)
        print("mri0 = ", mri.shape[0])
        running_test_acc = 100. * correct / len(test_loader.dataset)
        test_loss_mean = test_loss / len(test_loader.dataset)
        writer.add_scalar('testing loss/' + id, test_loss_mean, global_step=epoch)
        writer.add_scalar('testing accuracy/' + id, running_test_acc, global_step=epoch)

        test_loss_mean = test_loss / len(test_loader.dataset)
        print('\nTest {}: Average loss: {:.6f}, Accuracy: {}/{} ({:.4f}%)'.format(
            epoch, test_loss_mean, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
        print('\nTrue positive :{}, True negative :{}, False positive :{}, False negative :{}'.format(
            TP, TN, FP, FN))

    # Plotting Testing loss
    name = id
    plt.Figure(figsize=(13, 5))
    plt.title('Evolution of Loss curves')
    ax = plt.gca()
    # ax.set_ylim([-30, 30])
    plt.plot(running_train_acc, label='Training')
    plt.plot(running_test_acc, label='Testing')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc="upper right")
    plt.savefig(name)
