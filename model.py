import torch
from torch import nn
from torchsummary import summary


# 从已有的库中继承各种神经网络层，方便调用
class LeNet(nn.Module):
    # 初始化，定义准备网络层（卷积层、池化层）和激活函数
    def __init__(self):
        # 八股文
        super(LeNet, self).__init__()

        # 定义第一层卷积层
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2)
        # 定义激活函数
        self.sigmoid = nn.Sigmoid()
        # 定义池化
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        # 定义第二层卷积层
        self.conv3 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        # 定义池化
        self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
        # 展平，矩阵变一维向量
        self.flat = nn.Flatten()
        # 全连接层
        self.allconnect1 = nn.Linear(in_features=400, out_features=120)
        self.allconnect2 = nn.Linear(in_features=120, out_features=84)
        self.allconnect3 = nn.Linear(in_features=84, out_features=10)

    # 前向传播
    def forward(self, x):
        x = self.conv1(x)
        x = self.sigmoid(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.sigmoid(x)
        x = self.pool4(x)
        x = self.flat(x)
        x = self.allconnect1(x)
        x = self.sigmoid(x)
        x = self.allconnect2(x)
        x = self.sigmoid(x)
        x = self.allconnect3(x)
        return x


# 主函数（八股文）
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = LeNet().to(device)
    print(summary(model, input_size=(1, 28, 28)))
