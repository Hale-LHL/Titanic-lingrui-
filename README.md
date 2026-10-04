# Titanic-lingrui仓库说明

- 为方便进行两次提交的对比，特将一轮时提交的文件一并呈现
- 为方便阅读和复制等，学习笔记都上传了.pdf和.ipynb两种形式


> 更新说明:
> 本项目与一轮提交的相比,做出了以下调整:
> 1. 将数据处理分为dataset和dataloader(分区更有界限些)
> 2. 加入了config.py用以记录参数,路径以及各类列表
> 3. 将predict重分为了evaluate,predict,分别用以画图保存和预测新数据
> 4. 在1~3的基础上对每个板块进行了调整,让项目结构更清晰
> 5. 加入了L2正则方法
> 6. 多采用了ROC图像和混淆矩阵对模型进行评估
> 7. 加入对GPU的选择使用
> 8. 改善精确度和损失
> 9. 处理了Name特征,以其为基础增加了title的特征
> 10. 提交了多篇各板块学习笔记,对学习笔记进行了修改


- 文件结构如下
  
README.md

Titanic 2（二轮项目）    
├── data/     
│   └── titanic.csv     
├── docs/          
│   ├── image/      
│   │   ├── ROC_curve.png     
│   │   ├── accuracy_curve.png     
│   │   └── loss_curve.png     
│   └── learning_note/     
│       ├── matplotlib学习.ipynb     
│       ├── matplotlib学习.pdf      
│       ├── ndarray 学习.ipynb     
│       ├── ndarray 学习.pdf     
│       ├── numpy函数学习.ipynb     
│       ├── numpy函数学习.pdf      
│       ├── pandas学习.ipynb      
│       ├── pandas学习.pdf     
│       ├── titanic学习笔记.ipynb      
│       └── titanic学习笔记.pdf     
├── model/     
│   └── titanic.pth     
└── src/      
    ├── config.py         
    ├── dataloader.py         
    ├── dataset.py         
    ├── evaluate.py          
    ├── main.py          
    ├── model.py           
    ├── predict.py             
    ├── preprocess.py              
    ├── requirements.txt             
    └── train.py                

Titanic first（一轮项目）              
├── data/                 
│   └── titanic.csv               
├── docs/                 
│   ├── accuracy_curve.png             
│   ├── loss_curve.png              
│   └── titanic学习笔记.pdf             
├── src/                
    ├── data.py               
    ├── dataset.py               
    ├── main.py                 
    ├── model.py              
    ├── predict.py                 
    ├── preprocess.py           
    ├── requirements.txt                     
    └── train.py                 
             
