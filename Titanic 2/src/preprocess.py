

from src.config import STANDARDIZATION


def standardization_base(tr):
    mean = tr[STANDARDIZATION].mean()   #不能用全部的df，要用训练集，防止数据泄露
    std = tr[STANDARDIZATION].std().replace(0, 1.0)
    return mean,std

def standardization(mean,std,df):
    df[STANDARDIZATION] = (df[STANDARDIZATION] - mean) / std#这里的mean,std要按训练集定下来，不然预测时是一堆0
    return df

def drop_columns(df,DROP_COLUMNS):
    df = df.drop(columns=DROP_COLUMNS)
    return df


def fill( df,tr):
    df["Age"] = df["Age"].fillna(tr["Age"].median())
    df["Fare"] = df["Fare"].fillna(tr["Fare"].median())
    df["Embarked"] = df["Embarked"].fillna(tr["Embarked"].mode()[0])
    return df

def check(df):
    b = df.isnull().sum().astype(bool)  # isnull()：找缺失（False = 没有缺失）
    print('-----------------------------------')
    print('检查缺失值（False = 没有缺失）：')
    print(b.to_string())  # .to_string()：别给我省略号，整张表全打出来"
    print('检查类型（应全是数字）：')
    print(df.dtypes.to_string())
    print('-----------------------------------')
    return df


def recode(df):  # 独热编码
    df["Embarked_C"] = (df["Embarked"] == "C").astype(int)
    df["Embarked_Q"] = (df["Embarked"] == "Q").astype(int)
    df["Embarked_S"] = (df["Embarked"] == "S").astype(int)

    df["Pclass_1"] = (df["Pclass"] == 1).astype(int)
    df["Pclass_2"] = (df["Pclass"] == 2).astype(int)
    df["Pclass_3"] = (df["Pclass"] == 3).astype(int)

    df["Title_Miss"] = (df["Title"] == "Miss").astype(int)
    df["Title_Mrs"] = (df["Title"] == "Mrs").astype(int)
    df["Title_Master"] = (df["Title"] == "Master").astype(int)
    df["Title_Rare"] = (df["Title"] == "Rare").astype(int)

    df["Sex_male"] = (df["Sex"] == "male").astype(int)
    df = df.drop(columns=["Embarked", "Pclass", "Sex","Title"])#注意别重复删
    return df


def new_feature(df):
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
    #正则表达
    '''
    [A-Za-z]	   任意一个英文字母，大小写都行
    +	           前面那个字母可以连续好几个。所以 Mr、Mrs、Miss 全都能匹配上
    \.	           一个真正的句号。正则里光一个 . 是"任意字符"的通配符，加反斜杠才变回普通句号
    False          给回一列（Series），直接塞进 df 刚好；True 会给回一张表，还得再处理
    .extract()     按给的模板去每个名字里找，找到就拿出来，找不到填 NaN
    '''
    fixed = []
    for t in df["Title"]:
        if t in ["Mlle", "Ms"]:
            t = "Miss"
        if t == "Mme":
            t = "Mrs"
        if t == "Master":
            t = "Master"
        else:
            t = "Rare"
        fixed.append(t)

    df["Title"] = fixed
    df = df.drop(columns=["Name"])
    return df