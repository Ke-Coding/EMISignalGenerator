import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
import time
import os
from torchsummary import summary

# constants
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081
EPOCH_N = 6
BATCH_SIZE = 128
SAVE_PATH = './mbv2.pth'

using_gpu = torch.cuda.is_available()


class LinearBottleneck(nn.Module):
    def __init__(self, cin, k, cout, stride):
        super(LinearBottleneck, self).__init__()
        ex_ch = round(cin * k)
        self.block = nn.Sequential(
            nn.Conv2d(cin, ex_ch, kernel_size=1, padding=0, stride=1, bias=False),
            nn.BatchNorm2d(ex_ch),
            nn.ReLU6(),
            nn.Conv2d(ex_ch, ex_ch, kernel_size=3, padding=1, stride=stride, groups=ex_ch, bias=False),
            nn.BatchNorm2d(ex_ch),
            nn.ReLU6(),
            nn.Conv2d(ex_ch, cout, kernel_size=1, padding=0, stride=1, bias=False),
            nn.BatchNorm2d(cout)
        )
    
    def forward(self, x):
        return self.block(x)


class MBV2(nn.Module):
    def __init__(self, input_ch, num_classes, dropout_p, channels, last_ch, strides):
        super(MBV2, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_ch, channels[0], kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU6()
        )
        
        # Bottleneck第一层和第三层都是pointwise卷积，也即kernel_size=1，groups=1的卷积，
        # 第二层是depthwise卷积，也即kernel_size=3，groups=channels的卷积
        self.bottleneck1 = LinearBottleneck(cin=channels[0], k=3, cout=channels[1], stride=strides[0])
        self.bottleneck2 = LinearBottleneck(cin=channels[1], k=3, cout=channels[2], stride=strides[1])
        self.bottleneck3 = LinearBottleneck(cin=channels[2], k=3, cout=channels[3], stride=strides[2])
        
        self.last_ch = last_ch
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels[-1], self.last_ch, kernel_size=1, padding=0, stride=1, bias=False),
            nn.BatchNorm2d(self.last_ch),
            nn.ReLU6()
        )
        
        self.pool1 = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.using_dropout = dropout_p is not None
        if self.using_dropout:
            self.dropout = nn.Dropout(p=dropout_p)
        self.Dense = nn.Linear(self.last_ch, num_classes)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bottleneck1(out)
        out = self.bottleneck2(out)
        out = self.bottleneck3(out)
        out = self.conv2(out)
        out = self.pool1(out)
        out = out.view(-1, self.last_ch)
        if self.using_dropout:
            out = self.dropout(out)
        logits = self.Dense(out)
        return logits


def get_trainloader(batch_size):
    dataset = datasets.MNIST(root="./mmnist/", train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize(
                                     (MNIST_MEAN,), (MNIST_STD,)
                                 )
                             ]))
    return data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
    )


def get_testloader(batch_size):
    dataset = datasets.MNIST(root="./mmnist/", train=False, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize(
                                     (MNIST_MEAN,), (MNIST_STD,)
                                 )
                             ]))
    return data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,  # 每个epoch是否混淆
        num_workers=2,  # 多进程并发装载
        pin_memory=True,  # 是否使用锁页内存
        drop_last=False,  # 是否丢弃最后一个不完整的batch
    )


def train(train_data_loader, optimizer):
    epoch_acc = 0
    epoch_loss = 0.0
    train_dataset_length = 0
    tot_it = len(train_data_loader)
    last_time = time.time()
    for it, (x_train, y_train) in enumerate(train_data_loader):
        if using_gpu:
            x_train, y_train = x_train.cuda(), y_train.cuda()
        train_dataset_length += len(y_train)
        y_pred = model(x_train)
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(y_pred, y_train)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += torch.argmax(y_pred, dim=1).eq(y_train).sum().item()
        
        if it % 128 == 0:
            print(f'it: [{it}/{tot_it}],'
                  f' Loss: {epoch_loss:.4f}/{it + 1} = {epoch_loss / (it + 1):.4f},'
                  f' Acc: {epoch_acc}/{train_dataset_length} = {100 * epoch_acc / train_dataset_length:.3f}%,'
                  f' Iter time: {time.time() - last_time:.2f}s')
            last_time = time.time()
    
    return epoch_loss / tot_it, 100 * epoch_acc / train_dataset_length


def validation(test_data_loader):
    epoch_acc = 0
    epoch_loss = 0.0
    test_dataset_length = 0
    tot_it = len(test_data_loader)
    
    with torch.no_grad():
        model.eval()
        for it, (x_test, y_test) in enumerate(test_data_loader):
            test_dataset_length += len(y_test)
            if using_gpu:
                x_test, y_test = x_test.cuda(), y_test.cuda()
            y_pred = model(x_test)
            loss = nn.functional.cross_entropy(y_pred, y_test)
            epoch_loss += loss.item()
            epoch_acc += torch.argmax(y_pred, dim=1).eq(y_test).sum().item()
        model.train()
    return epoch_loss / tot_it, 100 * epoch_acc / test_dataset_length


model = MBV2(
    input_ch=1,
    num_classes=10,
    dropout_p=0.2,  # a scalar(probability) or None, if None: do not use dropout
    channels=[8, 12, 24, 32],
    last_ch=64,
    strides=[1, 2, 1],
)
if using_gpu:
    model = model.cuda()


def main():
    summary(model=model, input_size=(1, 28, 28))
    print(f'\n=== {["not using", "using"][using_gpu]} gpu ===')
    # pretrained_net = torch.load(PATH)
    # model.load_state_dict(pretrained_net)
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.005, momentum=0.9, weight_decay=1e-5, nesterov=True)
    # optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    train_data_loader = get_trainloader(batch_size=BATCH_SIZE)
    test_data_loader = get_testloader(batch_size=BATCH_SIZE)
    last_time = start_time = time.time()
    best_val_acc = 0
    for epoch in range(EPOCH_N):
        print(f'\n=== At epoch: [{epoch}/{EPOCH_N}] ===')
        train_loss, train_acc = train(train_data_loader=train_data_loader, optimizer=optimizer)
        val_loss, val_acc = validation(test_data_loader=test_data_loader)
        print(f'Epoch complete,'
              f' t-Loss: {train_loss:.4f}, t-Acc: {train_acc:.3f}%,'
              f' v-Loss: {val_loss:.4f}, v-Acc: {val_acc:.3f}%,'
              f' epoch time: {time.time() - last_time:.2f}s')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f'Best ckpt(acc {val_acc:.3f}) is saved at: {SAVE_PATH}')
            torch.save(model.state_dict(), SAVE_PATH)
        last_time = time.time()
    
    print(f'\n=== Training complete, best val_acc: {best_val_acc:.3f}%, total time: {time.time() - start_time:.2f}s ===')


if __name__ == '__main__':
    os.chdir('/content/drive/My Drive/mbv2_test')
    print('cwd: %s' % os.getcwd())
    
    main()
