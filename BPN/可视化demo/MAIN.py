import numpy as np
import os
from tensorflow.keras.models import model_from_json
import wfdb
from utils import qrs_detect, comp_cosEn, save_dict
import tensorflow as tf
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt



def load_model(path_weight , path_json):  # 导入数据和模型，对数据进行归一化处理，输出数据和模型
    with open(path_json, "r") as f:
        json_string = f.read()  # 读取本地模型的json文件
    model = model_from_json(json_string)  # 创建一个模型
    #model = fixnet.mix_layer((600,1))
    model.load_weights(path_weight)  # 导入模型参数

    return model


def load_data(sample_path):
    sig, fields = wfdb.rdsamp(sample_path)
    length = len(sig)
    fs = fields['fs']

    return sig, length, fs


def ngrams_rr(data, length):
    grams = []
    for i in range(0, length - 12, 12):
        grams.append(data[i: i + 12])
    return grams


def challenge_entry(sample_path):
    
    path_weight = r'the101171800th training record  of 0.h5'#导入模型参数
    path_json = r'fixresnet.json'#导入模型结构的文件，地址
    model = load_model(path_weight, path_json)#出来一个模型

    # model = unet((256, 1), 2, 0.001, maxpool=True, weights=None)
    # model.load_weights(path_weight)
    sig, _, fs = load_data(sample_path)#数据导入
    min_max_scaler = preprocessing.MinMaxScaler()  # 数据归一化处理
    sig = min_max_scaler.fit_transform(sig)
    # sig = sig[:, 1]
    outdata = np.zeros(2048)
    num_class = []
    num_class.append(1)
    # r_peaks = qrs_detect(sig, fs=200)
    co = 0
    x = [1,0]
    r_peaks = wfdb.rdann(sample_path, extension="atr").sample#导入官网下载的源文件
    former = 0
    for i, sample in enumerate(r_peaks):#对r波进行遍历
        if int(sample - 1024) < 0 :
            former +=1
            data_2leads = np.hstack((sig[int(0):int(1024), 0], sig[int(0):int(1024), 1]))
            outdata = np.vstack((outdata, data_2leads))
            continue
        if int(sample + 1024) > len(sig):
            former += 1
            data_2leads = np.hstack(
                (sig[int(len(sig) - 1024):int(len(sig)), 0], sig[int(len(sig) - 1024):int(len(sig)), 1]))
            outdata = np.vstack((outdata, data_2leads))
            continue
        data_2leads = np.hstack((sig[int(sample - 512):int(sample + 512),0],sig[int(sample - 512):int(sample + 512),1]))
        outdata = np.vstack((outdata, data_2leads))
    outdata = np.delete(outdata, 0, axis=0)
    val_dataset = tf.data.Dataset.from_tensor_slices((outdata))#构成数据集
    val_dataset = val_dataset.batch(len(r_peaks))
    for x_batch_val in val_dataset:#对数据集进行遍历
        val_logits = model(x_batch_val, training=False)#使用模型进行分类
        num_class = np.argmax(tf.nn.softmax(val_logits), 1)
#----------------------------------------------------------------------------------------------
    class_af = (np.sum(np.array(num_class) == 1) / len(num_class))

    return class_af,num_class


if __name__ == '__main__':
    # DATA_PATH = sys.argv[1]
    # RESULT_PATH = sys.argv[2]

    DATA_PATH = r'F:\毕业设计\icbeb2021-main\trainingI\training_I'
    RESULT_PATH = r'F:\毕业设计\imange\1'



    if not os.path.exists(RESULT_PATH):
        os.makedirs(RESULT_PATH)

    test_set = open(os.path.join(DATA_PATH, 'RECORDS'), 'r').read().splitlines()
    for i, sample in enumerate(test_set):
        #print(sample)
        sample_path = os.path.join(DATA_PATH, sample)
        class_af,num_class = challenge_entry(sample_path)
        plt.figure(i)
        plt.plot(num_class)
        #print('房颤比例：', class_af)
        plt.title('af :'+str(class_af))  # 标题
        plt.savefig('F:/毕业设计/imange/3/pic-{}.png'.format(i+1))
        #plt.show()
        plt.close()
        if class_af!=0:
            print(sample)
        #print(wfdb.rdheader(sample_path).comments)
