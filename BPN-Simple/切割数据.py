import numpy as np
import json
import os
import sys
from sklearn import preprocessing
import scipy.io as sio
import matplotlib.pyplot as plt
import wfdb
import re
from collections import Counter
length_ecg = 1024
#sample = 10
outdata = np.zeros(length_ecg*2)
outlabel = [0]
label = []
def original_reverse_diff(data, label,lens):
    #data = [[1, 2, 3], [-2, -3, -4], [4, 5, 6]]
    #label = [1, 0, 1, 1, 0, 0, 0, 1, 1]
    #n,m = np.shape(data)
    zeros_1 = np.ones(data.shape)*np.mean(data)
    reversedata = zeros_1 - data
    # zeros_2 = np.zeros((n)) diffdata = np.zeros(data.shape)
    #     diffdata[:,0:lens - 1] = np.diff(data, axis=1)
    #data_c = data[:,::-1]
    reversedata_c = reversedata[:,::-1]
    data = np.vstack((data,reversedata_c))
    label = np.hstack((label, label))
    return data,label
os.makedirs('./training_data')
os.makedirs('./test_data')
with open(r"F:\2019级\魏树鸿\21暑假\挑战赛\orgin data\fulldataset\RECORDS", "r") as f:
    dataname = f.readlines()
    print(dataname)
ccount = 0
cccount = 0
for line in dataname:
    line = line.strip('\n')  #去掉列表中每一个元素的换行符
    print(line)
    entity_startindex = [i.start() for i in re.finditer('_', line)]
    number = line[entity_startindex[0]+1:entity_startindex[1]]
    number = int(number)
    #count = Counter(ann.aux_note)
    #这个是数据集文件夹路径
    path = r"F:\2019级\魏树鸿\21暑假\挑战赛\orgin data\fulldataset\{}".format(line)
    ann = wfdb.rdann(path, extension="atr")  # 矫正数据大小删除最后一个心拍的后半段，也可以保留操作，但是这个使用的是删除操作，这样操作起来相对简单
    #break
    header = wfdb.rdheader(path)
    sig = wfdb.rdsamp(path)
    sig = sig[0]
    index_A = np.array([-1])#初始化数据
    min_max_scaler = preprocessing.MinMaxScaler()  # 数据归一化处理
    sig = min_max_scaler.fit_transform(sig) # 数据归一化处理
    #sig = sig[0:int((len(sig) / length_ecg)) * length_ecg, :]
    af_start_scripts = np.where((np.array(ann.aux_note) == '(AFIB') | (np.array(ann.aux_note) == '(AFL'))[0]
    af_end_scripts = np.where(np.array(ann.aux_note) == '(N')[0]#确定起始点的位置

    for i,af_end in enumerate(af_end_scripts):
        index_A = np.hstack((index_A,ann.sample[af_start_scripts[i]:af_end_scripts[i]]))
        #这里可以加一个进度条
    for index in ann.sample:
        if header.comments == ['paroxysmal atrial fibrillation'] and index not in index_A:
            continue
        data0 = np.hstack((sig[index - 512:index + 512, 0], sig[index - 512:index + 512, 1]))
        #data1 = np.hstack((sig[index - 200:index + 100, 0], sig[index - 200:index + 100, 1]))
        #data2 = np.hstack((sig[index - (150 + 100):index + (150 - 100), 0], sig[index - (150 + 100):index + (512 - 128), 1]))
        if data0.shape != (length_ecg*2,):# or data1.shape != (length_ecg*2,): #or data1.shape != (2048,) or data2.shape != (2048,):
            print('error')
            continue
        outdata = np.vstack((outdata, data0))
        #outdata = np.vstack((outdata, data1))
        #outdata = np.vstack((outdata, data2))
        if index in index_A:
            outlabel.append(1)
            #outlabel.append(1)
            #outlabel.append(1)
        else:
            outlabel.append(0)
            #outlabel.append(0)
            #outlabel.append(0)
        print(outdata.shape[0])
        if outdata.shape[0] > 1000:
            data = outdata
            label = outlabel
            idx = np.argwhere(np.all(data[:, ...] == 0, axis=1))
            label = np.delete(label, idx, axis=0)
            data = np.delete(data, idx, axis=0)
            data,label = original_reverse_diff(data,label,length_ecg)
            print('*' * 50)
            print('datashape:',data.shape,'labelshape:',label.shape)
            print('*' * 50)
            if number % 10 == 0:
                sio.savemat(
                    r'./training_data/aibi_{}(I)data.mat'.format(cccount),
                    {'datatrain': data})
                sio.savemat(
                    r'E:\数据\2019生理信号挑战赛testdata\aibi_{}(I)labell.mat'.format(cccount),
                    {'labeltrain': label})
                cccount = cccount + 1
            else:
                sio.savemat(
                    r'E:\数据\2019生理信号挑战赛traindata\aibi_{}(I)data.mat'.format(ccount),
                    {'datatrain': data})
                sio.savemat(
                    r'E:\数据\2019生理信号挑战赛traindata\aibi_{}(I)labell.mat'.format(ccount),
                    {'labeltrain': label})
                ccount = ccount + 1
            outdata = np.zeros(length_ecg*2)
            outlabel = [0]