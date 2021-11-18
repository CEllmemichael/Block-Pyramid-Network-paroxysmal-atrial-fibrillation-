import os
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf
from tensorflow.keras import optimizers

import numpy as np
import h5py
from tensorflow.keras.layers import Reshape, multiply
# import matplotlib.pyplot as plt
from load_data import LoadData
from tensorflow.keras.models import Model
from util import write_model_fix, loss_fn, ProgressBar
from model import mix_layer


def test_and_vild_acc(x_test, y_test):
    val_true = 0
    val_sum = 0
    TP_ = 0
    P = 0
    PP = 0
    batch_size = 300
    # Run a validation loop at the end of each epoch.
    val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    progress_bar = ProgressBar(int(x_test.shape[0] / batch_size) + 1)
    val_dataset = val_dataset.batch(batch_size)
    for i, [x_batch_val, y_batch_val] in enumerate(val_dataset):
        val_logits = model(x_batch_val, training=False)
        val_pre = np.argmax(tf.nn.softmax(val_logits), 1)
        val_pre = np.array([val_pre]).transpose()
        index = np.argwhere(np.where(y_batch_val == val_pre, 0, 1) == 0)
        y_batch_val_ = np.array(y_batch_val)
        val_true += index.shape[0]  # 所有正确的数量
        tp = np.sum(y_batch_val_[index[:, 0], index[:, 1]] == 0)  # 预测正确的里面属于第一类的数量
        p_ = np.sum(val_pre == 0)  # 预测中属于第一类的
        pp_ = np.sum(y_batch_val == 0)
        val_sum += (y_batch_val.shape[0] * y_batch_val.shape[1])
        TP_ += tp
        PP += pp_
        P += p_
        progress_bar.update(i)
    Precision = TP_ / P
    Recall = TP_ / PP
    val_acc = val_true / val_sum
    print('%' * 30)
    print("Acc: %.4f" % (float(val_acc),))
    print("Precision     : %.4f" % (float(Precision),))
    print("Recall        : %.4f" % (float(Recall),))
    print("Time taken    : %.2fs" % (time.time() - start_time))
    print('%' * 30)
    return val_acc


if __name__ == '__main__':
    batch_size = 512
    model = mix_layer((2048, 1))
    # write_model_fix(model)
    model.summary()
    # model.load_weights(r'/home/liutong/weishuhong/code1022/training_save_model/the104600th training record  of 0.h5')
    optimizer = optimizers.Adamax(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    epochs = 100
    step = 0

    summary_writer = tf.summary.create_file_writer('./tensorboard')  # 参数为记录文件所保存的目录
    # config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.9 # 占用GPU90%的显存
    # session = tf.Session(config=config)
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    # with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=True),
    #                 graph = detection_graph) as sess:
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # tf.compat.v1.disable_eager_execution()
    # hello = tf.constant('Hello,TensorFlow')
    # config = tf.compat.v1.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.9
    # sess = tf.compat.v1.Session(config=config)

    # gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    # tf.config.experimental.set_visible_devices(devices=gpus[0:4], device_type='GPU')
    if os.path.exists('result'):  # 如果文件存在
        os.remove('result')
    x_test1, y_test1 = LoadData(flag='test', parts=30)
    x_train, y_train, x_test, y_test = LoadData(flag='train', parts=50)
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

    # 在这里调节part,part分的越多，越能分出小的

    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch))
        start_time = time.time()
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            # step = step + steps
            with tf.GradientTape() as tape:
                logits = model(x_batch_train, training=True)
                # logits_array = np.array(logits)
                # logits_array[:, :, 0] = logits_array[:, :, 0]
                # #logits_array[:, :, 1] = logits_array[:, :, 1]
                # logits =tf.constant(logits_array)
                # logits[:, :, 0] = logits[:, :, 0] * 0.1
                loss_value = loss_fn(y_batch_train, logits)
            with summary_writer.as_default():  # 希望使用的记录器
                tf.summary.scalar("loss", loss_value, step=step)
            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            if step % 100 == 0:
                print(
                    "Training loss (for one batch) at step %d: %.4f"
                    % (step, float(loss_value))
                )
                print("Seen so far: %d samples" % ((step + 1) * batch_size))
                logits = np.array(tf.nn.softmax(logits))
                pre = np.argmax(tf.nn.softmax(logits), 1)
                pre = np.array([pre]).transpose()
                train_acc = len(np.where(y_batch_train - pre == 0)[0]) / (
                        y_batch_train.shape[0] * y_batch_train.shape[1])
                print("Training acc over epoch: %.4f" % (float(train_acc),))
                with open('result', "a") as file:
                    train_data_record = 'the' + str(epochs) + str(step) + 'th training record'
                    file.write(train_data_record + ':\n')
                    file.write('train_accuracy:' + str(train_acc) + '\n')
            #if step % 200 == 0 and step != 0:
        print('')
        print('Validation acc:')
        # x_test, y_test = load_data_directly_fun(0)
        val_acc = test_and_vild_acc(x_test, y_test)
        if val_acc > 0.5:
            try:
                os.makedirs('./training_save_model')
            except:
                pass
            train_data_record = './training_save_model/' + 'the' + str(epochs) + str(
                step) + 'th training record  of ' + str(epoch) + 'acc:' + str(val_acc) # +'wsh'
            Model.save(model, train_data_record + '.h5')
        with open('result.txt', "a") as file:
            train_data_record = 'the' + str(step) + 'th training record'
            file.write('Validation_accuracy:' + str(val_acc) + str(epochs) + str(
                step) + '\n')
            # if step % 1000 == 0 and step != 0:
        print('')
        print('Test acc:')
        val_acc = test_and_vild_acc(x_test1, y_test1)
        # del x_test1
        with open('result.txt', "a") as file:
            train_data_record = 'the' + str(step) + 'th training record'
            file.write('Test_accuracy:' + str(val_acc) + str(epochs) + str(
                step) + '\n')



