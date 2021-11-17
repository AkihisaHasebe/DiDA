import torch.nn as nn



class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.layer1_1 = self.module_CNN(1, 16)
        self.layer1_2 = self.module_CNN(16, 16)
        self.layer2_1 = self.module_CNN(16, 32)
        self.layer2_2 = self.module_CNN(32, 32)
        self.layer3_1 = self.module_CNN(32, 64)
        self.layer3_2 = self.module_CNN(64, 64)

        self.GAP = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(64,2)
        self.maxpool = nn.MaxPool2d(2)


    def forward(self, x):
        x = self.layer1_1(x)
        x = self.layer1_2(x)
        x = self.maxpool(x)
        
        x = self.layer2_1(x)
        x = self.layer2_2(x)
        x = self.maxpool(x)

        x = self.layer3_1(x)
        x = self.layer3_2(x)

        x = self.GAP(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x


    def module_CNN(self, input_channel, output_channel, kernel_size=3, padding=1):
        layer = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size= kernel_size, padding= padding),
            nn.BatchNorm2d(output_channel),
            nn.ReLU()
        )
        return layer