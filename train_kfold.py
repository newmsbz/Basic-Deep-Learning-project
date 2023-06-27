from torchvision.datasets import ImageFolder
import torch.utils.data as data
# from torch.utils.data import Subset
from torchvision import transforms, models
import torch
from torch.optim import lr_scheduler
from sklearn.model_selection import StratifiedKFold
import numpy as np

num_epochs = 100


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

num_folds = 5

dataset = ImageFolder("./dataset/images_original",
                      transform=transforms.Compose([transforms.RandomCrop(224),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]))

# Obtain the data and target labels
datas = [x[0] for x in dataset.samples]
targets = [x[1] for x in dataset.samples]

skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=805)

def train(model, criterion, optimizer, device, train_loader):    
    model.train()
    train_loss, correct, total = 0, 0, 0

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer_ft.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.long())
        loss.backward()
        optimizer_ft.step()

        train_loss += loss.data.cpu().numpy()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = train_loss /len(train_loader)
    epoch_acc = correct / total
    print('train | Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))


def test(model, criterion, test_loader):
    model.eval()
    test_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels.long())     

            test_loss += loss.data.cpu().numpy()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
                
        epoch_loss = test_loss / len(test_loader)
        epoch_acc = correct / total
        print('test | Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

# Iterate over the folds
for fold_idx, (train_index, test_index) in enumerate(skf.split(datas, targets)):
    print('------------------------------------------------------------')
    print('Fold {}/{}'.format(fold_idx + 1, num_folds))

    # Split the data into train and test sets for the current fold
    train_data = [datas[i] for i in train_index]
    train_targets = [targets[i] for i in train_index]
    test_data = [datas[i] for i in test_index]
    test_targets = [targets[i] for i in test_index]

    # Apply additional data augmentation
    train_transform = transforms.Compose([
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_imgs = data.Subset(dataset, train_index)
    train_imgs.dataset.transform = train_transform
    # Create the data loaders for the current fold
    # print(type(dataset), type(train_index))
    # train_index = torch.from_numpy(train_index)
    train_imgs = data.Subset(dataset, train_index)
    test_imgs = data.Subset(dataset, test_index)
    train_loader = data.DataLoader(train_imgs, batch_size=8, shuffle=True)
    test_loader = data.DataLoader(test_imgs, batch_size=8, shuffle=True)

    # Build and train the model
    resnet = models.resnet50(pretrained=True)
    num_ftrs = resnet.fc.in_features
    resnet.fc = torch.nn.Linear(num_ftrs, 10)
    resnet = resnet.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer_ft = torch.optim.Adam(filter(lambda p: p.requires_grad, resnet.parameters()), lr=0.001)

    for epoch in range(1, num_epochs + 1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        train(resnet, criterion, optimizer_ft, device, train_loader)
        test(resnet, criterion, test_loader)

torch.save(resnet, './save_weight/train_weight_{}.pt'.format(fold_idx))





# def main(model, criterion, optimizer, device, num_epochs=100):
#     for epoch in range(1, num_epochs + 1):
#         print('------------------------------------------------------------')
#         print('Epoch {}/{}'.format(epoch, num_epochs))
#         train(model, criterion, optimizer, device)
#         test(model, criterion)
#     torch.save(model, './save_weight/train_weight.pt')


# main(model=resnet, criterion=criterion, optimizer=optimizer_ft, device=device)