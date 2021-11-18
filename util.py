from tensorflow.python.keras.utils.np_utils import to_categorical
import tensorflow as tf
import sys
import numpy as np
def write_model_fix(model):
    json_string = model.to_json()
    with open(r'fixresnet.json', "w") as f:
        f.write(json_string)

def loss_fn(y_batch_train, logits):#计算损失函数
    y_train_onehot = to_categorical(y_batch_train)
    #    loss =tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels = y_train_onehot)
    loss = tf.keras.losses.categorical_crossentropy(y_train_onehot, logits)
    return tf.reduce_mean(loss)

# coding:utf-8


class ProgressBar():

    def __init__(self, max_steps):
        self.max_steps = max_steps
        self.current_step = 0
        self.progress_width = 50

    def update(self, step=None):
        self.current_step = step

        num_pass = int(self.current_step * self.progress_width / self.max_steps) + 1
        num_rest = self.progress_width - num_pass
        percent = (self.current_step+1) * 100.0 / self.max_steps
        progress_bar = '[' + '#' * (num_pass-1) + '->' + '-' * num_rest + ']'
        progress_bar += '%.2f' % percent + '%'
        if self.current_step < self.max_steps - 1:
            progress_bar += '\r'
        else:
            progress_bar += '\n'
        sys.stdout.write(progress_bar)
        sys.stdout.flush()
        if self.current_step >= self.max_steps:
            self.current_step = 0
            print

def CreatIndexList(index = [70,100]):
    out = []
    index = np.array(index).reshape(-1,2).astype(int)
    for (start,end) in index:
        for i in range(start,end+1):
            out.append(i)
    return out


