import pandas as pd
import torch

from src.config import ORDER, DROP_COLUMNS
from src.preprocess import new_feature, drop_columns, fill, recode, standardization
#还可以在preprocess里面设一个类，这边就可以只导入类，但这样导入函数更清晰


def predict(tr,mean,std,mt):#mt,pp两个对象不用引入到这个文件，在main中给他就是了

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    m = input("是否开始预测（是，否）：")
    if m == "是":
        print('开始预测')

        m = input('是否使用默认数据（一等舱25岁女性）？填 是 或 否：')

        if m == '是':
            PassengerId = 999
            Pclass = 1
            Name = 'Demo, Miss. Lily'
            Sex = 'female'
            Age = 25.0
            SibSp = 0
            Parch = 0
            Ticket = 'UNKNOWN'
            Fare = 100.0
            Cabin = None
            Embarked = 'S'
            print('已使用默认乘客：Demo, Miss. Lily，一等舱25岁')
        else:
            PassengerId = int(input('PassengerId 乘客编号（随便填，不影响结果）：'))
            Pclass = int(input('Pclass 舱位（填 1 或 2 或 3）：'))
            Name = input('Name 姓名（可选Miss. Lily）：')
            Sex = input('Sex 性别（填 male 或 female）：')
            Age = float(input('Age 年龄：'))
            SibSp = int(input('SibSp 同行的兄弟姐妹配偶数：'))
            Parch = int(input('Parch 同行的父母子女数：'))
            Ticket = input('Ticket 票号（随便填，不影响结果）：')
            Fare = float(input('Fare 票价：'))
            Cabin = input('Cabin 舱位号（随便填，不影响结果）：')
            Embarked = input('Embarked 登船港（填 C 或 Q 或 S）：')

        raw = {'PassengerId': PassengerId, 'Survived': None, 'Pclass': Pclass,
               'Name': Name, 'Sex': Sex, 'Age': Age, 'SibSp': SibSp,
               'Parch': Parch, 'Ticket': Ticket, 'Fare': Fare, 'Cabin': Cabin,
               'Embarked': Embarked}

        new_df = pd.DataFrame([raw])
        new_df = new_feature(new_df)
        new_df = drop_columns(new_df, DROP_COLUMNS)
        new_df = fill(new_df, tr)
        new_df = recode(new_df)
        new_df = standardization(mean,std,new_df)
        new_one = new_df[ORDER].values.astype('float32')

        mt.eval()
        with torch.no_grad():
            out = mt(torch.tensor(new_one).to(device))
            prodi = torch.softmax(out, dim=1)[0, 1].item()

        if prodi > 0.5:
            result = '幸存'
        else:
            result = '遇难'

        if Sex == 'male':
            sex = '男'
        else:
            sex = '女'

        print('%s等舱%s岁%s性预测%s，幸存概率为 %.4f' % (Pclass, Age, sex, result, prodi))
        print("预测结束")

    else:
        print("任务结束")

    return 0

