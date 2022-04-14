import torch
import torch.nn.functional as F


def train(model, train_loader, optimizer, epoch):
    # model.train().to(device)
    total_loss = 0
    for batch_idx, (mri, label) in enumerate(train_loader):
        # mri, pet, label = mri.to(device), pet.to(device), label.to(device)
        optimizer.zero_grad()
        # data = (mri.type(torch.FloatTensor), mri.type(torch.FloatTensor), label.type(torch.FloatTensor))
        # data = (mri.type(torch.FloatTensor), label.type(torch.FloatTensor))
        data = (mri.type(torch.FloatTensor))
        output = model(data)
        loss = F.nll_loss(output, label, reduction='mean')
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        if batch_idx % 5 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * 5, len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

    return total_loss


def test(model, test_loader, epoch):
    # model.eval().to(device)
    test_loss = 0
    correct = 0
    # out_list = []
    # pred_list = []
    # label_list = []
    with torch.no_grad():
        for mri, label in test_loader:
            # mri, pet, label = mri.to(device), pet.to(device), label.to(device)
            # data = (mri.type(torch.FloatTensor), mri.type(torch.FloatTensor), label.type(torch.FloatTensor))
            # data = (mri.type(torch.FloatTensor), label.type(torch.FloatTensor))
            data = (mri.type(torch.FloatTensor))
            output = model(data)
            test_loss += F.nll_loss(output, label, reduction='mean').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(label.view_as(pred)).sum().item()

            # out_list.append(output.item())
            # pred_list.append(pred.item())
            # label_list.append(label.item())

    test_loss /= len(test_loader.dataset)
    print('\nTest {}: Average loss: {:.6f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        epoch, test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return test_loss
