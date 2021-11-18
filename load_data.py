import numpy as np
from sklearn.model_selection import train_test_split
import os
import scipy.io as sio
from util import ProgressBar
import torch
def LoadData(flag = 'train', parts = 50):
    if flag == 'train':
        list_lens = 1800
        test_size = 0.1
        rootpath = r'E:\数据\2019生理信号挑战赛traindata'#此处修改文件夹的路径和文件的数目
    elif flag == 'test':
        list_lens = 180
        test_size = 0.000001
        rootpath = r'E:\数据\2019生理信号挑战赛testdata'

    index = list(range(list_lens))#Creat a number list used for next step of shuffle

    index = index[0:int(len(index) / parts) * parts]#remove some data ending of the list so,the length of list can be divided by parts(args)
    np.random.shuffle(index)#creat the random list between different batches
    index = np.array(index)
    index = index.reshape(parts, -1)# In this martrix every lines index used for loading data onces

    data_length = 2048
    full_data = np.zeros((data_length))
    full_label = np.zeros((2000))


    for part in range(parts):
        print('loading {} data of part{} ....'.format(flag, part))
        index_part = index[part,:]
        progress_bar = ProgressBar(len(index_part))
        #break
        for ii,number in enumerate(index_part):
            #break
            path_data = r'{}/aibi_{}(I)data.mat'.format(rootpath,number)
            path_label = r'{}/aibi_{}(I)labell.mat'.format(rootpath,number)
            abspath_data = os.path.abspath(path_data)
            abspath_label = os.path.abspath(path_label)
            feature_data = sio.loadmat(abspath_data)
            feature_label = sio.loadmat(abspath_label)
            data = feature_data['datatrain']
            label = feature_label['labeltrain']
            if label.shape[1] != data.shape[0] or data.shape[1] != data_length or label.shape[1] != 2000:
                continue
            full_data = np.vstack((full_data, data))
            full_label = np.vstack((full_label, label))
            progress_bar.update(ii)
        full_data = np.delete(full_data, 0, 0)
        full_label = np.delete(full_label, 0, 0)
        full_label = full_label.reshape(-1, 1)
        x_train, x_test, y_train, y_test = train_test_split(full_data, full_label, test_size=test_size, random_state=5)
        full_data = np.zeros((data_length))
        full_label = np.zeros((2000))
        if part == 0:
            x_trains = x_train
            y_trains = y_train
            x_tests = x_test
            y_tests = y_test
        else:
            x_trains = np.vstack((x_trains,x_train))
            y_trains = np.vstack((y_trains,y_train))
            x_tests = np.vstack((x_tests, x_test))
            y_tests = np.vstack((y_tests, y_test))
    x_train = np.array(x_trains, dtype=np.float32)
    y_train = np.array(y_trains, dtype=np.long)
    x_test = np.array(x_tests, dtype=np.float32)
    y_test = np.array(y_tests, dtype=np.long)
    # Train_data = []
    # for j in range(len(y_train)):
    #     Train_data.append([x_train[j], y_train[j]])
    if flag == 'test':
        return x_train, y_train
    if flag == 'train':
        return x_train,y_train,x_test,y_test#Train_data


def LoadDataTorch(flag = 'train', parts = 50):
    if flag == 'train':
        list_lens = 1800
        test_size = 0.1
        rootpath = r'E:\数据\2019生理信号挑战赛traindata'#此处修改文件夹的路径和文件的数目
    elif flag == 'test':
        list_lens = 180
        test_size = 0.000001
        rootpath = r'E:\数据\2019生理信号挑战赛testdata'

    index = list(range(list_lens))#Creat a number list used for next step of shuffle

    index = index[0:int(len(index) / parts) * parts]#remove some data ending of the list so,the length of list can be divided by parts(args)
    np.random.shuffle(index)#creat the random list between different batches
    index = np.array(index)
    index = index.reshape(parts, -1)# In this martrix every lines index used for loading data onces

    #data_length = 2048
    # full_data = np.zeros((data_length))
    # full_label = np.zeros((2000))


    for part in range(parts):
        print('loading {} data of part{} ....'.format(flag, part))
        index_part = index[part,:]
        progress_bar = ProgressBar(len(index_part))
        break
        for ii,number in enumerate(index_part):
            break
            path_data = r'{}/aibi_{}(I)data.mat'.format(rootpath,number)
            path_label = r'{}/aibi_{}(I)labell.mat'.format(rootpath,number)
            abspath_data = os.path.abspath(path_data)
            abspath_label = os.path.abspath(path_label)
            feature_data = sio.loadmat(abspath_data)
            feature_label = sio.loadmat(abspath_label)
            data = feature_data['datatrain']
            data = np.array(data, dtype=np.float32)
            label = feature_label['labeltrain']
            label = np.array(label, dtype=np.long)
            if label.shape[1] != data.shape[0]:#or data.shape[1] != data_length or label.shape[1] != 2000:
                continue
            if ii == 0:
                full_data = data
                full_label = label
            else:
                full_data = np.vstack((full_data, data))
                full_label = np.vstack((full_label, label))
                progress_bar.update(ii)
        # full_data = np.delete(full_data, 0, 0)
        # full_label = np.delete(full_label, 0, 0)
        full_label = full_label.reshape(-1, 1)
        x_train, x_test, y_train, y_test = train_test_split(full_data, full_label, test_size=test_size, random_state=5)
        # full_data = np.zeros((data_length))

        # full_label = np.zeros((2000))
        Train_data = []
        for j in range(len(y_train)):
            Train_data.append([x_train[j], y_train[j]])
        Test_data = []
        for j in range(len(y_test)):
            Test_data.append([x_train[j], y_train[j]])
        if part == 0:
            Train_data = np.array(Train_data)
            Test_data = np.array(Test_data)
            Train_datas = torch.from_numpy(Train_data).cuda()
            Test_datas = torch.from_numpy(Test_data).cuda()
        else:
            Train_data = np.array(Train_data)
            Test_data = np.array(Test_data)
            Train_datas = torch.cat((Train_datas,Train_data),1)
            Test_datas = torch.cat((Test_datas, Test_data), 1)
    # Train_data = []
    # for j in range(len(y_train)):
    #     Train_data.append([x_train[j], y_train[j]])
    if flag == 'test':
        return x_train, y_train
    if flag == 'train':
        return x_train,y_train,x_test,y_test#Train_data




