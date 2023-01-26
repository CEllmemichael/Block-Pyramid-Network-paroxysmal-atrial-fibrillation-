属于心电信号分析技术领域，包括心电信号预处理模块，一维卷积双交叉臂网络模块，房颤心拍风险提示模块以及外部接口模块。心电信号预处理模块用于对输入的心电信号进行预处理；一维卷积双交叉臂网络模块用于对双导联特征进行特征融合和选择；房颤心拍风险提示模块用于将输入的单导联心电信号特征输入到神经网络中，得到房颤心拍的置信度；外部接口模块用于兼容不同心电采集设备和报警输出接口。本发明还公开了一种包括上述系统的房颤风险提示设备。采用本发明的房颤风险提示系统，通过对两个导联的心拍进行融合并抽取特征，实现对双导联心拍进行高质量的房颤检测，有效提高双导联心拍的房颤检测准确率。

![图片](https://user-images.githubusercontent.com/66575985/214832535-e28b1ea8-a8f4-4fdf-8cea-2ee60db298ce.png)

![图片](https://user-images.githubusercontent.com/66575985/214831977-3d3cdd13-2436-45fc-928c-95a600f8d550.png)
![图片](https://user-images.githubusercontent.com/66575985/214832089-928c40f0-ed99-4f12-9f2e-dd7feb10415a.png)
![图片](https://user-images.githubusercontent.com/66575985/214832182-7a441338-dd84-4cf0-b159-be48c1e88170.png)
![图片](https://user-images.githubusercontent.com/66575985/214832259-13f26163-db8b-46df-a7b2-c66b5c32a4a3.png)
![图片](https://user-images.githubusercontent.com/66575985/214832410-df7c1a0e-78c2-40b1-a4d0-7d1958a06875.png)
![图片](https://user-images.githubusercontent.com/66575985/214832451-365863cc-fc5c-477d-8a03-d035225b59e2.png)
实例：

![图片](https://user-images.githubusercontent.com/66575985/214832901-f04b2c86-78fd-4c6f-8d61-479b46eb0994.png)
系统输出：

![图片](https://user-images.githubusercontent.com/66575985/214832949-dd05d47b-4c3b-49f7-bdab-e8f3a61f8c38.png)

