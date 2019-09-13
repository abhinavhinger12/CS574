
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.nn import functional as F


class LogisticRegression(nn.Module):
    def __init__(self, input_size, inter,num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)
    
    def forward(self, x):
        out = F.sigmoid(self.linear(x))
        return out


class MLP1linear(nn.Module):
    def __init__(self, input_size, inter,num_classes):
        super(MLP1linear, self).__init__()
        self.fc1 = nn.Linear(input_size, inter)
        self.fc2 = nn.Linear(inter, num_classes)
    
    def forward(self, x):
        out = self.fc2(self.fc1(x))
        return out
      
class MLP1relu(nn.Module):
    def __init__(self, input_size, inter,num_classes):
        super(MLP1relu, self).__init__()
        self.fc1 = nn.Linear(input_size, inter)
        self.fc2 = nn.Linear(inter, num_classes)
    
    def forward(self, x):
        out = self.fc2(F.relu(self.fc1(x)))
        return out

class MLP1relu2(nn.Module):
    def __init__(self, input_size, inter,num_classes):
        super(MLP1relu2, self).__init__()
        self.fc1 = nn.Linear(input_size, inter)    
        self.dropper = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(inter, num_classes)
    
    def forward(self, x):
        out = self.fc2(self.dropper(F.relu(self.fc1(x))))
        return out

      
class MLP1relu3(nn.Module):
    def __init__(self, input_size, inter,num_classes):
        super(MLP1relu3, self).__init__()
        self.fc1 = nn.Linear(input_size, inter)    
        self.dropper = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(inter, num_classes)
    
    def forward(self, x):
        out = self.fc2(self.dropper(F.relu(self.fc1(x))))
        return out
      

class MLP1relu5(nn.Module):
    def __init__(self, input_size, inter,num_classes):
        super(MLP1relu5, self).__init__()
        self.fc1 = nn.Linear(input_size, inter)    
        self.dropper = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(inter, num_classes)
    
    def forward(self, x):
        out = self.fc2(self.dropper(F.relu(self.fc1(x))))
        return out
      
class MLP1tanh(nn.Module):
    def __init__(self, input_size, inter,num_classes):
        super(MLP1tanh, self).__init__()
        self.fc1 = nn.Linear(input_size, inter)
        self.fc2 = nn.Linear(inter, num_classes)
    
    def forward(self, x):
        out = self.fc2(F.tanh(self.fc1(x)))
        return out
      
class MLP1sig(nn.Module):
    def __init__(self, input_size, inter,num_classes):
        super(MLP1tanh, self).__init__()
        self.fc1 = nn.Linear(input_size, inter)
        self.fc2 = nn.Linear(inter, num_classes)
    
    def forward(self, x):
        out = self.fc2(F.sigmoid(self.fc1(x)))
        return out

      
class MLP2linear(nn.Module):
    def __init__(self, input_size, inter,num_classes):
        super(MLP2linear, self).__init__()
        self.fc1 = nn.Linear(input_size, 300)
        self.fc2 = nn.Linear(300, inter)
        self.fc3 = nn.Linear(inter, num_classes)
    

    def forward(self, x):
        out = self.fc3(self.fc2(self.fc1(x)))
        return out

class MLP3linear(nn.Module):
    def __init__(self, input_size, inter,num_classes):
        super(MLP2linear, self).__init__()
        self.fc1 = nn.Linear(input_size, 300)
        self.fc2 = nn.Linear(300, inter)
        self.fc3 = nn.Linear(inter, num_classes)

    def forward(self, x):
        out = self.fc4(self.fc3(self.fc2(self.fc1(x))))
        return out



class CNN1c(nn.Module):
    def __init__(self, input_size, inter,num_classes):
        super(CNN1c, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.fc1 = nn.Linear(1440, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = x.view(-1, 1440)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x)

class CNN2c(nn.Module):
    def __init__(self, input_size, inter,num_classes):
        super(CNN2c, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x)


class CNN1ck3(nn.Module):
    def __init__(self, input_size, inter,num_classes):
        super(CNN1ck3, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3)
        self.fc1 = nn.Linear(1690, 320)
        self.fc2 = nn.Linear(320, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = x.view(-1, 1690)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x)
    
class CNN1ck5(nn.Module):
    def __init__(self, input_size, inter,num_classes):
        super(CNN1ck5, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.fc1 = nn.Linear(1440, 320)
        self.fc2 = nn.Linear(320, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = x.view(-1, 1440)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x)

class CNN1ck7(nn.Module):
    def __init__(self, input_size, inter,num_classes):
        super(CNN1ck7, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=7)
        self.fc1 = nn.Linear(1210, 320)
        self.fc2 = nn.Linear(320, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = x.view(-1, 1210)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x)

class CNN2cLinear(nn.Module):
    def __init__(self, input_size, inter,num_classes):
        super(CNN2cLinear, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = (F.max_pool2d(self.conv1(x), 2))
        x = (F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = (self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x)

class CNN2cTanh(nn.Module):
    def __init__(self, input_size, inter,num_classes):
        super(CNN2cTanh, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.tanh(F.max_pool2d(self.conv1(x), 2))
        x = F.tanh(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.tanh(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x)

class CNN2cSigmoid(nn.Module):
    def __init__(self, input_size, inter,num_classes):
        super(CNN2cSigmoid, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.sigmoid(F.max_pool2d(self.conv1(x), 2))
        x = F.sigmoid(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x)

class CNN1cNK5(nn.Module):
    def __init__(self, input_size, inter,num_classes):
        super(CNN1cNK5, self).__init__()
        self.conv1 = nn.Conv2d(1, 5, kernel_size=5)
        self.fc1 = nn.Linear(720, 320)
        self.fc2 = nn.Linear(320, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = x.view(-1, 720)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x)

class CNN1cNK10(nn.Module):
    def __init__(self, input_size, inter,num_classes):
        super(CNN1cNK10, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.fc1 = nn.Linear(1440, 320)
        self.fc2 = nn.Linear(320, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = x.view(-1, 1440)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x)

class CNN1cNK20(nn.Module):
    def __init__(self, input_size, inter,num_classes):
        super(CNN1cNK20, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5)
        self.fc1 = nn.Linear(2880, 320)
        self.fc2 = nn.Linear(320, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = x.view(-1, 2880)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x)
