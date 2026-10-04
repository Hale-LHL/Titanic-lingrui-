import torch.nn as nn

from src.config import H1,H2,DROPOUT


class Net(nn.Module):
    def __init__(self, in_dim, h1=H1, h2=H2, dropout=DROPOUT):
        super().__init__()#继承了父类nn.Module就要写这一行
        self.net = nn.Sequential(#Sequential用来打包后面的东西
            nn.Linear(in_dim, h1),
            nn.ReLU(),#不用Sigmoid（实测）
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, 2), )#二分类输出为2

    def forward(self, x):  #相当于调用nn.Sequential（），并输出结果
        return self.net(x)
