'''config用来存放参数，地址等等
   修改数据可以在这里完成'''

import os

PATH_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PATH_ROOT,"data","titanic.csv")
ACCURACY_CURVE_PATH = os.path.join(PATH_ROOT,"docs","image",'accuracy_curve.png')
LOSS_CURVE_PATH =os.path.join(PATH_ROOT,"docs","image",'loss_curve.png')
ROC_CURVE_PATH = os.path.join(PATH_ROOT,"docs","image",'ROC_curve.png')
MODEL_PATH = os.path.join(PATH_ROOT, "model", "titanic.pth")

STANDARDIZATION = ['Age', 'Fare', 'SibSp', 'Parch', 'FamilySize']         #要标准化的特征

ORDER = [
    'Age', 'Fare', 'SibSp', 'Parch', 'FamilySize', 'IsAlone',
    'Sex_male',
    'Pclass_1', 'Pclass_2', 'Pclass_3',
    'Embarked_C','Embarked_Q', 'Embarked_S',
    "Title_Miss","Title_Mrs","Title_Rare","Title_Master",]   #特征顺序

DROP_COLUMNS =["Ticket", "Cabin"]

TITLE_COLS = ["Master", "Miss", "Mr", "Mrs", "Rare"]

TEST_SIZE =0.2
SEED = 42
EPOCHS = 200
LR = 0.01
H1 = 32
H2 = 16
DROPOUT = 0.2
WEIGHT_DECAY = 0.5