import os
import torch
from matplotlib import pyplot as plt
from src.config import ORDER, ACCURACY_CURVE_PATH, LOSS_CURVE_PATH, PATH_ROOT, MODEL_PATH, \
    ROC_CURVE_PATH  # 函数外的值直接导import，函数里的用return
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

def draw(losses, test_accs,train_accs,y_true, y_score):
#-----------画LOSS曲线--------------------------
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig(LOSS_CURVE_PATH, dpi=120)
    plt.close()  #要关掉，不然要覆盖
#-----------画accs曲线--------------------------
    plt.plot(test_accs,label='test',color = "orange",linestyle=':')
    plt.plot(train_accs,label='train',color = "blue")
    plt.legend()

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy')

    plt.savefig(ACCURACY_CURVE_PATH, dpi=120)
    plt.close()

#-----------画ROC曲线--------------------------
    #fpr：误报率数组 → 后面当横轴
    #pr：抓对率数组 → 后面当纵轴
    #th：门槛数组，留着挑最佳门槛用的
    fpr, tpr, th = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    cm = confusion_matrix(y_true, (y_score > 0.5).astype(int))#混淆矩阵只认 0/1，不认布尔
    print('AUC =', auc)
    print("混淆矩阵为：")
    print(cm)
    plt.figure()
    plt.plot(fpr, tpr, color='red', lw=2, label='Model (AUC = %.3f)' % auc)
    plt.plot([0, 1], [0, 1], '--', color='gray', label='Random (AUC = 0.5)')#第一个[0, 1]两个点的横坐标，第二个[0, 1]两个点的纵坐标
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc=0)
    plt.savefig(ROC_CURVE_PATH, dpi=120)
    plt.close()
    print('曲线已保存')

def save_model(mt):
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)# exist_ok=True允许已存在
    torch.save(mt.state_dict(), MODEL_PATH)
    print('模型已保存到', MODEL_PATH)



