# 本工程文件为BNN网络训练模块
其中：

# .py文件
Date_process_CPSC_DATA为对CPSC数据集进行预处理，在其中输入数据集所在地址，对数据集进行切分，数据增强和格式转换的工作。
Date_process_MIT_DATA为对MIT数据集进行预处理，在其中输入数据集所在地址，对数据集进行切分，数据增强和格式转换的工作。
Load_data为分批次向显卡读入数据，防止显存不够导致训练结束，被Train.py调用
Model为模型文件，被Train.py调用
Train为模型训练主函数
util为训练过程小脚本函数存放处，被Train.py调用
requirements.txt为本机python环境包

# dir文件
tensorboard为存放训练过程中训练参数文件夹
test_data为预处理代码（Date_process_CPSC_DATA或Date_process_MIT_DATA）自动生成存放测试数据文件夹
training_data为预处理代码自动生成存放训练数据文件夹
training_save_model为训练过程存放训练参数文件夹

# 调用过程：
S1:使用 Date_process_CPSC_DATA或Date_process_MIT_DATA生成.mat数据（需要输入源数据所在地址）

S2:使用Train对网络进行训练
