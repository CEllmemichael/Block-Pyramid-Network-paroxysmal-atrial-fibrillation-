# Block Pyramid Network 
---
本系统包括四个模块：心电信号预处理模块、一维卷积双交叉臂网络模块、房颤心拍风险提示模块和外部接口模块。其中，心电信号预处理模块用于对输入的心电信号进行预处理；一维卷积双交叉臂网络模块用于对双导联特征进行特征融合和选择；房颤心拍风险提示模块用于将输入的单导联心电信号特征输入到神经网络中，得到房颤心拍的置信度；外部接口模块用于兼容不同心电采集设备和报警输出接口。此外，本系统还包括一个房颤风险提示设备，包括上述四个模块。采用本系统的房颤风险提示系统，通过对两个导联的心拍进行融合并抽取特征，能够实现对双导联心拍进行高质量的房颤检测，有效提高双导联心拍的房颤检测准确率。


The system includes four modules: the electrocardiogram (ECG) signal preprocessing module, the one-dimensional convolution double-cross arm network module, the atrial fibrillation (AF) heartbeat risk alert module, and the external interface module. The ECG signal preprocessing module is used to preprocess the input ECG signal; the one-dimensional convolution double-cross arm network module is used to fuse and select features of the dual-lead characteristics; the AF heartbeat risk alert module is used to input the single-lead ECG signal features into a neural network to obtain the confidence of AF heartbeat; the external interface module is used to be compatible with different ECG acquisition devices and alarm output interfaces. In addition, the system also includes an AF risk alert device that includes the above four modules. By fusing and extracting features from the heartbeats of two leads, the AF risk alert system using this system can achieve high-quality AF detection of dual-lead heartbeats, effectively improving the AF detection accuracy of dual-lead heartbeats.
## 竞赛题目：The 4th China Physiological Signal Challenge 2021 (CPSC 2021) aims to encourage the development of algorithms for searching the paroxysmal atrial fibrillation (PAF) events from dynamic ECG recording
网址：（http://2021.icbeb.org/CPSC2021）
+ Task I  : Three classification tasks for the whole ECG signal
对于整条信号的三分类任务：全部房颤心拍的信号，全部正常心拍的信号，房颤阵法型心拍信号
+ Task II : Endpoint recognition tasks for paroxysmal atrial fibrillation signal
对于阵法型房颤心拍产生的端点的识别

![图片](https://user-images.githubusercontent.com/66575985/214835438-0fd93a8f-de9a-4d9b-a6ee-90617c2da94e.png)
## 数据预处理 Data processing

> Two ways for augmentation processing

1:Differential processing

a:The difference is greater

2:The center mirror image processing

a:The difference is greater

b:Reserve the noise component of the original signal
![图片](https://user-images.githubusercontent.com/66575985/214837961-fabb81da-6800-4497-9442-73039ab90384.png)

## 本项目的方案 Game ideas
+ Part I : Cut out
![图片](https://user-images.githubusercontent.com/66575985/214836746-9c8f3907-a8da-4b0c-a5e3-f1a7f906c93d.png)
+ Part II : Sew

Normal：

No five consecutive heart beats are recognized as atrial fibrillation

Atrial Fibrillation：

Not five consecutive heart beats are recognized as normal

Paroxysmal Atrial Fibrillation：

There are five consecutive heart beats with atrial fibrillation and normal![图片](https://user-images.githubusercontent.com/66575985/214837081-fd40f4c6-efa0-44ac-b1c2-10ffffc446b8.png)












## 网络模型 Neural network design
![图片](https://user-images.githubusercontent.com/66575985/214839244-700740f9-472c-4c28-860d-8948e641d22a.png)
> Fusion layer  makes better use of the two leads data provided by the competition



## 其中一个模块实例 example network moudle：

![图片](https://user-images.githubusercontent.com/66575985/214832901-f04b2c86-78fd-4c6f-8d61-479b46eb0994.png)

## 系统输出效果演示 output figure：

![图片](https://user-images.githubusercontent.com/66575985/214832949-dd05d47b-4c3b-49f7-bdab-e8f3a61f8c38.png)

