import numpy as np
from torch.utils.data import DataLoader

from src.config import TEST_SIZE, SEED, ORDER
from src.dataset import TitanicDataset


def split(df, test_size=TEST_SIZE, seed=SEED):
    #设置种子
    np.random.seed(seed)
    #转类型
    df_survived = df['Survived'].values.astype('int64')
    #分为是否幸存
    surv = np.where(df_survived == 1)[0]  # [0]才能得到元组中的下标
    dead = np.where(df_survived == 0)[0]
    #第一次打乱
    np.random.shuffle(surv)
    np.random.shuffle(dead)
    #分训练集和测试集
    s_te_idx = int(len(surv) * test_size)
    d_te_idx = int(len(dead) * test_size)  # 注意转成整型
    te = np.concatenate([surv[:s_te_idx], dead[:d_te_idx]])
    tr = np.concatenate([surv[s_te_idx:], dead[d_te_idx:]])  # 外层也有个[]
    #第二次打乱
    np.random.shuffle(te)
    np.random.shuffle(tr)
    return te, tr

def dataloader(df,df_raw,tr,te):
    x = df[ORDER].values.astype('float32')
    y = df_raw['Survived'].values.astype('int64')

    #标号配对
    xtr, ytr = x[tr], y[tr]
    xte, yte = x[te], y[te]

    print('训练集', len(xtr), '条, 测试集', len(xte), '条')
    print('训练集幸存率', round(ytr.mean(), 3), ', 测试集幸存率', round(yte.mean(), 3))

    train_dataset = TitanicDataset(xtr, ytr)
    test_dataset = TitanicDataset(xte, yte)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)#train_dataset是对象，不是变量
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

    return train_loader,test_loader,yte,len(ORDER)
