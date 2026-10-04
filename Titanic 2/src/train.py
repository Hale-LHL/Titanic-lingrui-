import random

import numpy as np
import torch
import torch.nn as nn
from src.model import Net #各个文件之间可以通过import来交接类或函数
from src.config import EPOCHS, LR, SEED, WEIGHT_DECAY


def set_seed(seed = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # 固定种子

    torch.cuda.manual_seed_all(seed)  # 给所有 GPU 设种子
    torch.backends.cudnn.deterministic = True  # 卷积等算子走确定性算法
    torch.backends.cudnn.benchmark = False  # 关掉"自动挑最快算法"（它会带来不确定性）
    return 0

def train(in_dim,train_loader, test_loader, yte, epochs=EPOCHS, lr=LR,seed=SEED):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("你使用的设备是：","GPU" if torch.cuda.is_available() else "CPU")
    mt = Net(in_dim)
    mt = mt.to(device)

    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(mt.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    #AdamW 比 Adam 更适合配 weight_decay（Adam 会把惩罚和自适应步长搅在一起）
    losses, test_accs,train_accs = [], [],[]

    for epoch in range(epochs):
        mt.train()#切换训练模式，主要是开启dropout
        total_loss, total_n = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = mt(xb)#算输出
            loss = loss_fn(out, yb)

            opt.zero_grad()#清除梯度
            loss.backward()#反向传播算梯度
            opt.step()#正式工作

            total_loss += loss.item() * len(yb)#.item（）用来把张量转成数字 (画图也会用)
            total_n += len(yb)

        losses.append(total_loss / total_n)

        mt.eval()#关闭训练模式，主要是关闭dropout（不然测试集结果不准）
        with torch.no_grad():#with自动资源释放,不用在意开关
            correct, total = 0, 0
            for xb, yb in test_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                predi = mt(xb).argmax(1)#挑出每个乘客分数更高的那一类，得到 0 或 1（分数是模型的输出，不是概率，没有和为一，要知道概率，用softmax）
                correct += (predi == yb).sum().item()
                total += len(yb)
            test_acc = correct / total
            test_accs.append(test_acc)

        with torch.no_grad():#with自动资源释放,不用在意开关
            correct, total = 0, 0
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                predi = mt(xb).argmax(1)#挑出每个乘客分数更高的那一类，得到 0 或 1（分数是模型的输出，不是概率，没有和为一，要知道概率，用softmax）
                correct += (predi == yb).sum().item()
                total += len(yb)
            train_acc = correct / total
            train_accs.append(train_acc)

        if (epoch + 1) % 20 == 0 or epoch + 1 ==1:
            print('第', epoch + 1, '轮 | 损失', round(losses[-1], 4),'| 测试准确率', round(test_acc, 4))#保留4位小数

    print('最终测试准确率:', round(test_accs[-1], 4))#【-1】是最后一次
    print('盲猜遇难的概率:', round((yte == 0).mean(), 4))

    return mt, losses, test_accs,train_accs

def score(mt, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mt.eval()
    p_all = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            out = mt(xb)
            p = torch.softmax(out, dim=1)[:, 1]
            p_all.append(p.cpu().numpy())
    return np.concatenate(p_all)
#concatenate 的作用就是把外层那个列表拆掉，把里面的数组原样吐出来，变成一条货真价实的 177 长数组

