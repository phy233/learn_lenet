import copy
import time

import torch
from torchvision.datasets import FashionMNIST
from torchvision import transforms
import torch.utils.data as data
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from model import LeNet


def train_and_verification_data_process():
    # 下载数据集，同时将灰度图归一化并转换为张量
    train_data = FashionMNIST(root='./data',
                              train=True,
                              transform=transforms.Compose([transforms.Resize(size=28), transforms.ToTensor()]),
                              download=True)

    # 划分训练集和验证机
    train_data, val_data = data.random_split(train_data,
                                             lengths=[round(0.8 * len(train_data)), round(0.2 * len(train_data))])
    # 加载训练集
    train_data_loaded = data.DataLoader(dataset=train_data,
                                        batch_size=64,
                                        shuffle=True,
                                        num_workers=0)
    # 加载验证集
    verfication_data_loaded = data.DataLoader(dataset=train_data,
                                              batch_size=64,
                                              shuffle=True,
                                              num_workers=0)

    return train_data_loaded, verfication_data_loaded


# 模型训练
def train_model_process(model, train_data_loaded, verfication_data_loaded, num_epoch):
    # 选择训练设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 定义优化器（更厉害的梯度下降法），用于更新参数，学习率为0.001
    # PyTorch 提供的 Adam 优化器（一种常用的自适应学习率优化算法，结合了 Momentum 和 RMSprop 的优点，收敛稳定且高效）
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 定义损失函数（交叉熵损失函数）
    critersion = nn.CrossEntropyLoss()

    # 模型放入训练设备
    model = model.to(device)

    # 复制模型当前参数
    best_model_wts = copy.deepcopy(model.state_dict())

    # 初始化一些参数
    # 准确率
    best_acc = 0.0
    # 训练集的loss值
    train_loss_all = []
    # 验证集的loss值
    verification_loss_all = []
    # 训练集的准确度
    train_acc_all = []
    # 验证集的准确度
    verification_acc_all = []
    # 当前时间，计算训练时间
    since = time.time()

    # 开始训练
    for epoch in range(num_epoch):
        # 打印当前轮次信息
        print("current epoch:{},total epoch:{}".format(epoch, num_epoch - 1))
        print("-" * 10)

        # 初始化参数
        # 初始化本轮loss
        train_loss = 0.0
        # 初始化本轮准确度
        train_corrects = 0

        # 验证集的
        verification_loss = 0.0
        verification_corrects = 0

        # 训练和验证的样本数量
        train_num = 0
        verification_num = 0

        # 一批一批取数据，对每一捆进行训练和计算
        # 提取每一捆的数据和标签，bx为128*28*28，by为128*10
        for step, (b_x, b_y) in enumerate(train_data_loaded):
            # 丢到设备中
            b_x = b_x.to(device)
            b_y = b_y.to(device)

            # 开启训练模式，开始训练
            model.train()

            # 数据丢到模型里面开始计算，输出128*10
            output = model(b_x)

            # 将输出转换为概率，并找到对应概率最大值的行标
            pre_label = torch.argmax(output, dim=1)

            # 计算loss
            # 损失函数需要原始输出的 “概率分布信息”
            # 模型的原始输出（如output）通常是未归一化的分数（logits），或经过 Softmax 等激活后的概率分布
            # 损失函数（如分类任务中常用的CrossEntropyLoss）的计算依赖于这种概率分布信息，而非单一的预测标签：
            # 例如，交叉熵损失会比较模型输出的概率分布（output）与真实标签的 “one-hot 分布”（b_y对应的独热编码），通过两者的差异计算损失。
            # 这种差异能反映模型对每个类别的 “置信度”，而不仅仅是 “是否预测正确”，从而为参数更新提供更丰富的梯度信息。
            # argmax后的标签是离散值，无法计算梯度
            # pre_label = torch.argmax(output, dim=1)的结果是离散的类别索引（例如[3, 5, 0, ...]），它是对原始输出的 “硬决策”（只保留概率最大的类别）。
            # 如果用pre_label计算损失，会导致两个问题：
            # 丢失概率信息：离散标签无法反映模型对其他类别的置信度（例如，模型对类别 A 的预测概率是 90% 还是 51%，对损失计算的意义完全不同）。
            # 梯度无法传播：argmax是一个 “不可导” 的操作（函数在跳跃点的导数为 0 或不存在），如果损失基于pre_label计算，反向传播时无法得到有效的梯度，导致模型参数无法更新。
            # 损失函数的内部处理逻辑
            # 以 PyTorch 的CrossEntropyLoss为例，它的输入就是模型的原始输出（logits），内部会自动完成：
            # 对 logits 应用 Softmax，得到概率分布；
            # 计算与真实标签的交叉熵损失。
            # 这也是为什么不需要先手动对output做 Softmax，直接用原始输出计算损失即可 —— 损失函数已经包含了这一步骤。
            loss = critersion(output, b_y)

            # 准备梯度下降，将梯度置为0，因为pytorch计算梯度是自动累加的，在计算下一批次时要对上一轮梯度清零
            optimizer.zero_grad()

            # 反向传播，参数更新
            loss.backward()
            optimizer.step()

            # 累计当前批次的总损失（用于后续计算平均损失）。
            # loss.item()：将 PyTorch 张量（loss）转换为 Python 数值（避免计算图占用内存）。
            # b_x.size(0)：当前批次的样本数量（batch_size）。
            # 因为loss通常是批次平均损失（默认情况下，CrossEntropyLoss等会对批次内样本取平均），所以乘以batch_size得到批次总损失，再累加得到整个 epoch 的总损失。
            train_loss += loss.item() * b_x.size(0)

            # 累计当前批次中预测正确的样本数量（用于后续计算准确率）
            # pre_label == b_y.data：逐元素比较预测标签（pre_label）和真实标签（b_y），返回布尔张量（True表示预测正确）。
            # torch.sum(...)：统计布尔张量中True的数量（即当前批次正确预测数）。
            # train_corrects累加所有批次的正确数，最终用于计算整个 epoch 的准确率（正确数 / 总样本数）
            train_corrects += torch.sum(pre_label == b_y.data)

            # 统计已经训练的样本数量
            train_num += b_x.size(0)

        # 验证数据
        for step, (b_x, b_y) in enumerate(verfication_data_loaded):
            # 丢到验证设备中
            b_x = b_x.to(device)
            b_y = b_y.to(device)

            # 模型进入评估模式
            model.eval()

            # 前向传播，计算预测结果
            output = model(b_x)

            # 将输出转换为概率，并找到对应概率最大值的行标，也即获取正确的类别
            pre_label = torch.argmax(output, dim=1)

            # 计算loss
            loss = critersion(output, b_y)

            # 累计当前批次的总损失（用于后续计算平均损失）。
            verification_loss += loss.item() * b_x.size(0)
            # 累计当前批次中预测正确的样本数量（用于后续计算准确率）
            verification_corrects += torch.sum(pre_label == b_y.data)
            # 统计已经验证的样本数量
            verification_num += b_x.size(0)

        # 计算本轮训练的相关值
        # 存储每一次训练产生的loss，因为train_loss是每一批次训练累加的，所以在处理该轮训练的时候，要除以这一轮总的参与训练的总数
        train_loss_all.append(train_loss / train_num)
        # correct值转双精度，因为分子分母的值类型都是int
        train_acc_all.append(train_corrects.double().item() / train_num)

        # 验证集也一样
        verification_loss_all.append(verification_loss / verification_num)
        verification_acc_all.append(verification_corrects.double().item() / verification_num)

        # 打印指标
        # epoch：当前训练轮次（如第 1 轮、第 10 轮），表示模型已完整遍历训练集的次数。
        # train_loss_all[-1]：train_loss_all 是存储每轮训练集总损失（或平均损失）的列表，[-1] 取列表最后一个元素，即当前轮次的训练损失。
        # train_acc_all[-1]：train_acc_all 是存储每轮训练集准确率的列表，[-1] 取当前轮次的训练准确率。
        # {:.4f}：格式化输出，保留 4 位小数，让指标显示更整齐、易读（如损失值 2.34567 会显示为 2.3457）。
        print('{} Train loss: {:.4f}, Train acc:{:.4f}'.format(epoch, train_loss_all[-1], train_acc_all[-1]))
        print('{} Verification loss: {:.4f}, Verification acc:{:.4f}'.format(epoch, verification_loss_all[-1],
                                                                             verification_acc_all[-1]))

        # 保留最高准确度及其相关参数
        if verification_acc_all[-1] > best_acc:
            best_acc = verification_acc_all[-1]
            # 保存当前参数
            best_model_wts = copy.deepcopy(model.state_dict())

        # 本轮结束，计算用时
        time_use = time.time() - since
        print("本轮训练用时：{:.0f}分{:.0f}秒".format(time_use / 60, time_use % 60))

    # 选择最优参数
    # 保存状态字典：
    # 仅保存模型的参数（推荐方式），加载时需先定义模型结构，再通过 model.load_state_dict(torch.load(...)) 恢复参数，兼容性更强（不受模型类定义变化影响）
    torch.save(best_model_wts, 'D:/My_Program/Python/machine_learning/lenet/best_model.pth')

    # 保存相关数据
    train_process = pd.DataFrame(data={"epoch": range(num_epoch),
                                       "train_loss_all": train_loss_all,
                                       "verification_loss_all": verification_loss_all,
                                       "train_acc_all": train_acc_all,
                                       "verification_acc_all": verification_acc_all,
                                       })

    return train_process


# 画图
def matplot_acc_loss(train_process):
    plt.figure(figsize=(12, 4))
    # 第一张图，画loss
    plt.subplot(1, 2, 1)
    plt.plot(train_process["epoch"], train_process.train_loss_all, 'ro-', label="train loss")
    plt.plot(train_process["epoch"], train_process.verification_loss_all, 'bs-', label="verification loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    # 第二张图，画acc
    plt.subplot(1, 2, 2)
    plt.plot(train_process["epoch"], train_process.train_acc_all, 'ro-', label="train acc")
    plt.plot(train_process["epoch"], train_process.verification_acc_all, 'bs-', label="verification acc")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("acc")

    plt.show()


# 主函数
if __name__ == "__main__":
    # 模型实例化
    lenet = LeNet()
    # 获取数据
    train_data_loaded, verfication_data_loaded = train_and_verification_data_process()
    # 开始训练
    train_process = train_model_process(lenet, train_data_loaded, verfication_data_loaded, num_epoch=20)
    # 画图
    matplot_acc_loss(train_process)
