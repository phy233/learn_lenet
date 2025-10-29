import numpy as np
import torch
import torch.utils.data as data
from matplotlib import pyplot as plt
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from model import LeNet
import seaborn as sns
from sklearn.metrics import confusion_matrix


def test_data_process():
    # 下载数据集，同时将灰度图归一化并转换为张量
    test_data = FashionMNIST(root='./data',
                             train=False,
                             transform=transforms.Compose([transforms.Resize(size=28), transforms.ToTensor()]),
                             download=True)

    # 加载训练集
    test_data_loaded = data.DataLoader(dataset=test_data,
                                       batch_size=1,
                                       shuffle=True,
                                       num_workers=0)

    return test_data_loaded


def test_model_process(model, test_data_loaded):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 模型放到设备当中
    model = model.to(device)

    # 初始化准确度
    test_correct = 0.0
    # 初始化测试的样本总数
    test_num = 0

    # 关闭梯度计算，因为推理不涉及到反向传播，加快运行速度
    with torch.no_grad():
        for test_data_x, test_data_y in test_data_loaded:
            # 将特征和标签都放入设备中
            test_data_x = test_data_x.to(device)
            test_data_y = test_data_y.to(device)

            # 模型设为评估模式
            model.eval()
            # 前向传播得到预测结果
            output = model(test_data_x)

            # 将输出结果通过argmax函数转换成概率，并获取最大可能的标签
            # dim=1，指定沿着第一维度（列）寻找最大值
            pre_lab = torch.argmax(output, dim=1)

            # 计算正确率
            test_correct += torch.sum(pre_lab == test_data_y.data)
            # 累加测试样本数量
            test_num += test_data_x.size(0)

    # 计算准确率
    test_acc = test_correct.double().item() / test_num
    print("correct:", test_acc)


def test_with_detail(model, test_data_loaded):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 模型放到设备当中
    model = model.to(device)

    classes = FashionMNIST.classes
    result_all = []
    label_all = []

    with torch.no_grad():
        for bx, by in test_data_loaded:
            bx = bx.to(device)
            by = by.to(device)

            model.eval()
            output = model(bx)
            prelabel = torch.argmax(output, dim=1)
            # 张量转数值
            result = prelabel.item()
            label = by.item()

            print("预测值：", classes[result], "真实值：", classes[label])
            result_all.append(result)
            label_all.append(label)

    return result_all, label_all


def plot_confusion_matrix(y_true, y_pred, class_names,
                          figsize=(10, 8),
                          cmap=plt.cm.Blues):
    """
    绘制混淆矩阵

    参数:
    - y_true: 真实标签（一维数组，如 [0, 1, 2, ...]）
    - y_pred: 预测标签（一维数组，与 y_true 长度相同）
    - class_names: 类别名称列表（如 ['猫', '狗', '鸟']）
    - figsize: 图像大小
    - cmap: 颜色映射
    """
    # 设置中文字体（关键步骤）
    plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei"]
    # 解决负号显示问题（可选）
    plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)

    # 归一化（可选，将数值转换为比例，范围 0-1）
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # 创建画布
    plt.figure(figsize=figsize)
    ax = plt.subplot(111)

    # 绘制热力图
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap=cmap,
                xticklabels=class_names, yticklabels=class_names,
                ax=ax)

    # 设置标签和标题
    plt.xlabel('预测类别')
    plt.ylabel('真实类别')
    plt.title('混淆矩阵（归一化）')

    # 调整布局并显示
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 加载模型
    model = LeNet()
    model.load_state_dict(torch.load("best_model.pth"))

    # 加载测试数据
    test_data_loaded = test_data_process()

    # 加载模型测试
    # test_model_process(model, test_data_loaded)
    result_all, label_all = test_with_detail(model, test_data_loaded)
    plot_confusion_matrix(y_true=label_all, y_pred=result_all, class_names=FashionMNIST.classes)
