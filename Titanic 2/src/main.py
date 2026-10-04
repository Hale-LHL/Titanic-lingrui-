import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'   #防止torch 与 numpy 的 OpenMP打架
from src.config import DATA_PATH, DROP_COLUMNS
from src.evaluate import draw, save_model
from src.predict import predict
from src.preprocess import new_feature, fill, recode, standardization, standardization_base, check, drop_columns
from src.train import set_seed, train, score
from src.dataloader import split, dataloader



import pandas as pd

#--------------------固定种子----------------------------
set_seed()
#--------------------读取数据----------------------------
df = pd.read_csv(DATA_PATH)
#--------------------复制数据----------------------------
df_raw = df.copy()#全程的数据修改在df上完成，df_raw为备份
#--------------------观察数据----------------------------
print(df.info())
#--------------------删除无效特征----------------------------
df =drop_columns(df,DROP_COLUMNS)
#--------------------数据集划分----------------------------
te,tr = split(df)
#--------------------创造特征----------------------------
df = new_feature(df)
#--------------------填缺失值----------------------------
tr_df = df.iloc[tr]          # 用行号把训练集那批人真正取出来
fill(df, tr_df)
#--------------------独热编码----------------------------
df = recode(df)
#--------------------结果检查----------------------------
check(df)
#--------------------标准化----------------------------
mean,std = standardization_base(tr_df)
standardization(mean,std,df)
#--------------------建dataloader----------------------------
train_loader, test_loader, yte,IN_DIM = dataloader(df,df_raw,tr,te)
#--------------------开始训练----------------------------
mt,losses, test_accs,train_accs = train(IN_DIM,train_loader, test_loader, yte)
#--------------------计算数据----------------------------
y_score = score(mt, test_loader)
#--------------------结果处理----------------------------
draw(losses, test_accs,train_accs,yte,y_score)
#--------------------保存模型----------------------------
save_model(mt)
#--------------------进行预测----------------------------
predict(tr_df,mean,std,mt)