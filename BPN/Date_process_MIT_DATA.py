import os
import wfdb
from sklearn import preprocessing
import numpy as np
import scipy.io as sio

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

'''
处理数据到两个文件夹
'''
if __name__ == '__main__':
    DATA_PATH = r'E:\数据\mit-bih-atrial-fibrillation-database-1.0.0\files'
    try:
    os.makedirs('./training_data')
    os.makedirs('./test_data')
    except:
        pass
    length_ecg = 1024
    count = 0
    ccount = 0
    flag = 0
    data_set = open(os.path.join(DATA_PATH, 'RECORDS'), 'r').read().splitlines()
    for data_name in data_set:
        sample_path = os.path.join(DATA_PATH, data_name)
        number = int(data_name[len(data_name) - 1])
        try:
            sig = wfdb.rdsamp(sample_path)
            sig = sig[0]
            min_max_scaler = preprocessing.MinMaxScaler()  # 数据归一化处理
            sig = min_max_scaler.fit_transform(sig)  # 数据归一化处理
            print(len(sig))
        except:
            continue
        ann = wfdb.rdann(sample_path, extension="atr")
        index_note = ann.sample
        index_note = np.hstack(([0],index_note,len(sig)))
        note = ann.aux_note
        header = wfdb.rdheader(sample_path)
        R_peaks = wfdb.rdann(sample_path, extension="qrs").sample
        note_j = 0
        outlabel = []
        for ii,index in enumerate(R_peaks):
            if not(index > index_note[note_j] and index < index_note[note_j+1]):
                note_j += 1
            if ann.aux_note[note_j-1] == '(N':
                label = 0
            else:
                label = 1
            data = np.hstack((sig[index - 512:index + 512, 0], sig[index - 512:index + 512, 1]))
            if data.shape != (length_ecg * 2,):
                continue

            if ii == 0 or flag == 0 :
                outdata = data
                flag = 1
                flags = 0
            else:
                outdata = np.vstack((outdata, data))
                flags = 1
            outlabel.append(label)
            if outdata.shape[0] > 1000 and flags == 1:
                if number < 2:#% 2 == 0:
                    sio.savemat(
                        r'./test_data/aibi_{}(I)data.mat'.format(count),
                        {'datatrain': outdata})
                    sio.savemat(
                        r'./test_data/aibi_{}(I)labell.mat'.format(count),
                        {'labeltrain': outlabel})
                    count += 1
                else:
                    outdata, outlabel = original_reverse_diff(outdata, outlabel, length_ecg)
                    sio.savemat(
                        r'./train_data/aibi_{}(I)data.mat'.format(ccount),
                        {'datatrain': outdata})
                    sio.savemat(
                        r'./train_data/aibi_{}(I)labell.mat'.format(ccount),
                        {'labeltrain': outlabel})
                    ccount += 1
                outlabel = []
                flag = 0
