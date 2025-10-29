from torchvision.datasets import FashionMNIST
from torchvision import transforms
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt

# 下载数据集，同时将灰度图归一化并转换为张量
train_data = FashionMNIST(root='./data',
                          train=True,
                          transform=transforms.Compose([transforms.Resize(size=28 * 28), transforms.ToTensor()]),
                          download=True
                          )

# 加载数据集，打包64一捆，打乱顺序
train_load = data.DataLoader(dataset=train_data,
                             batch_size=64,
                             shuffle=True,
                             num_workers=0
                             )

# 获取每一捆数据及其对应标签
# 通过enumerate遍历训练数据加载器（train_load，通常是DataLoader对象），每次迭代返回一个批次的数据。
# step是迭代步数（从 0 开始），(b_x, b_y)是一个元组，其中b_x是该批次的输入数据（如图像、序列），b_y是对应的标签（如分类任务的类别）。
for step, (b_x, b_y) in enumerate(train_load):
    if step > 0:
        break

# 将四维张量移除第一维（通道数），并转换为numpy数组（矩阵）
# 通过squeeze()移除维度为 1 的轴（例如，若输入数据有冗余的单通道维度[batch, 1, height, width]，会被压缩为[batch, height, width]）
batch_x = b_x.squeeze().numpy()
# 标签也转换成numpy格式
batch_y = b_y.numpy()
# 训练集总的标签
class_label = train_data.classes
print(class_label)

# 展示一捆数据
# 创建一个新的绘图窗口（figure），设置窗口尺寸为宽 12 英寸、高 5 英寸，确保批量图像显示时不会过于拥挤
plt.figure(figsize=(12, 5))

# 遍历当前批次的所有样本，len(batch_y)是批次大小（即该批次包含的样本数量），np.arange生成从 0 到 “批次大小 - 1” 的索引，用于定位每个样本
for i in np.arange((len(batch_y))):

    # 在绘图窗口中创建子图网格：
    # 第一个参数4：子图总行数；
    # 第二个参数16：子图总列数；
    # 第三个参数i + 1：当前子图的位置（从 1 开始计数，i从 0 开始，故加 1）。
    # 该设置最多可显示 4×16=64 个样本，若批次大小小于 64，剩余位置会留白。
    plt.subplot(4, 16, i + 1)

    # 显示第i个样本的图像：
    # batch_x[i, :, :]：取第i个样本的图像数据，其形状为(高度, 宽度)（已通过之前的squeeze()移除单通道维度）；
    # cmap=plt.cm.gray：使用灰度色彩映射，表明这是灰度图像。
    plt.imshow(batch_x[i, :, :], cmap=plt.cm.gray)

    # 为当前子图添加标题（即图像的类别标签）：
    # batch_y[i]：第i个样本的标签索引（如 0 对应 “cat”、1 对应 “dog”）；
    # class_label[...]：通过标签索引获取具体类别名称（如class_label[0]为 “0 - zero”）；
    # size=10：设置标题字体大小为 10，避免文字过大遮挡图像。
    plt.title(class_label[batch_y[i]], size=10)

    # 关闭当前子图的坐标轴（包括 x 轴、y 轴的刻度和边框）
    plt.axis("off")

    # 调整子图之间的水平间距（wspace）为 0.05，减小图像间的空白
    plt.subplots_adjust(wspace=0.05)

# 显示整个绘图窗口，呈现所有子图组成的批量图像网格
plt.show()
