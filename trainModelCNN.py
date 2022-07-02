import torch
import torch.nn.functional as F
import torch.nn as nn


def train(model, train_loader, optimizer, epoch, loss_fn, device):
    model.train()
    total_loss = 0
    correct = 0
    for batch_idx, (mri, label) in enumerate(train_loader):
        mri, label = mri.to(device), label.to(device)
        # label = label.unsqueeze(1)
        data = (mri.type(torch.FloatTensor), label.type(torch.FloatTensor))

        output = model(data)
        loss = F.nll_loss(output, label, reduction='mean')

        # Checking batch
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(label.view_as(pred)).sum().item()

        # loss = nn.BCELoss()
        # label = label.to(torch.float32)
        # loss = loss_fn(output, label)
        # correct += output.eq(label.view_as(output)).sum().item()
        # pred = (output > 0.5).float()

        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # if batch_idx % 10 == 0:
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * 10, len(train_loader.dataset),
                   100. * batch_idx / len(train_loader), loss.item()))
        print("Correct = ", correct)

    return total_loss


def test(model, test_loader, epoch, loss_fn, device):
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
            test_loss += F.nll_loss(output, label, reduction='mean').item()

            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(label.view_as(pred)).sum().item()
            truths = pred.eq(label.view_as(pred))

            # label = label.to(torch.float32)
            # test_loss += loss_fn(output, label).item()
            # pred = (output > 0.5).float()
            # correct += pred.eq(label.view_as(pred)).sum().item()

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

    test_loss_mean = test_loss / len(test_loader.dataset)
    print('\nTest {}: Average loss: {:.6f}, Accuracy: {}/{} ({:.4f}%)'.format(
        epoch, test_loss_mean, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    print('\nTrue positive :{}, True negative :{}, False positive :{}, False negative :{}'.format(
        TP, TN, FP, FN))

    return test_loss
