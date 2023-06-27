from torchvision.datasets import ImageFolder
import torch.utils.data as data
from torchvision import transforms, models
import torch
from torch.optim import lr_scheduler

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

train_imgs = ImageFolder("./dataset/images_train",
                         transform=transforms.Compose([transforms.RandomCrop(224),
                                                       transforms.ToTensor(),
                                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]))

test_imgs = ImageFolder("./dataset/images_test",
                        transform=transforms.Compose([transforms.RandomCrop(224),
                                                      transforms.ToTensor(),
                                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]))

train_loader = data.DataLoader(train_imgs, batch_size=8, shuffle=True)
test_loader = data.DataLoader(test_imgs, batch_size=8, shuffle=True)

# print(train_loader)
# for batch in zip(train_loader, test_loader):
#     print(batch)


resnet = models.resnet50(pretrained=True)
num_ftrs = resnet.fc.in_features # fc는 모델의 마지막 layer를, in_features는 해당 층의 입력 채널 수 반환
resnet.fc = torch.nn.Linear(num_ftrs, 10) # 마지막 fc층의 출력 채널을 클래스 수에 맞게 변환
resnet = resnet.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer_ft = torch.optim.Adam(filter(lambda p : p.requires_grad, resnet.parameters()), lr=0.001)

# exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size = 7, gamma = 0.1)

def train(model, criterion, optimizer, device):    
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


def test(model, criterion):
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


def main(model, criterion, optimizer, device, num_epochs=300):
    for epoch in range(1, num_epochs + 1):
        print('------------------------------------------------------------')
        print('Epoch {}/{}'.format(epoch, num_epochs))
        train(model, criterion, optimizer, device)
        test(model, criterion)
    torch.save(model, './save_weight/train_weight.pt')


main(model=resnet, criterion=criterion, optimizer=optimizer_ft, device=device)