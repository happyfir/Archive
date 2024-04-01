import pandas as pd
import time
import numpy as np
import warnings
#模型相关
import tensorflow as tf
import keras
from keras.models import load_model
from keras.layers import Input, BatchNormalization, Conv1D, Multiply, Concatenate, Bidirectional, LSTM, Embedding, \
    Lambda, Add, Conv2D, Subtract, Average
from keras.layers import GlobalMaxPooling1D, Dense, Activation, Dropout
from keras.callbacks import EarlyStopping
from keras.backend.tensorflow_backend import set_session
from sklearn.model_selection import StratifiedKFold
#画图相关
import os

import model.model

os.environ["PATH"] += os.pathsep + 'D:/SoftWare/Graphviz2.38/bin/'
import seaborn as sns
sns.set_style('white')
from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
# specify which GPU will be used
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#config = tf.ConfigProto()
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
#set_session(tf.Session(config=config))
# set_session(tf.compat.v1.Session(config=config))
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

#超参数
max_length = 1000

class ClassifyGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, list_IDs, datasets, labels, batch_size=32, dim=max_length, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.datasets = datasets
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.zeros((self.batch_size, self.dim, 102), dtype=float)
        X2 = np.zeros((self.batch_size, self.dim, 12), dtype=float)
        y = np.zeros(self.batch_size, dtype=float)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            base_path = "../Dataset/dataset/dataset_15000/{0}.npy"
            item = self.datasets.iloc[ID]
            self_name = item['file_name']
            tmp = np.load(base_path.format(self_name))
            tmp = np.clip(tmp, -100, 100)
            if tmp.shape[0] > self.dim:
                X[i] = tmp[:self.dim, :]
                X2[i] = tmp[:self.dim, :12]
            else:
                X[i, :tmp.shape[0], :] = tmp[:, :]
                X2[i, :tmp.shape[0], :] = tmp[:, :12]
            y[i] = self.labels[ID]
        return X, y


class Model():
    def __init__(self):
        self.start_time = time.time()

    def get_model(self, model_path=None):
        if model_path is None:
            self.params_input = Input(shape=(max_length, 102), name="input_layer")  # (?,1000,102)

            # 原来的模型
            self.x = BatchNormalization(name="batch_normalization_1")(self.params_input)

            self.x_0 = Conv1D(128, 2, strides=1, padding='same')(self.x)
            self.x_1 = Conv1D(128, 2, strides=1, activation="sigmoid", padding='same')(self.x)
            self.gated_0 = Multiply()([self.x_0, self.x_1])

            self.y_0 = Conv1D(128, 3, strides=1, padding='same')(self.x)
            self.y_1 = Conv1D(128, 3, strides=1, activation="sigmoid", padding='same')(self.x)
            self.gated_1 = Multiply()([self.y_0, self.y_1])

            self.k_0 = Conv1D(128, 4, strides=1, padding='same')(self.x)
            self.k_1 = Conv1D(128, 4, strides=1, activation="sigmoid", padding='same')(self.x)
            self.gated_2 = Multiply()([self.k_0, self.k_1])

            self.cat_gate = Concatenate()([self.gated_0, self.gated_1, self.gated_2])
            self.inputData = BatchNormalization(name="batch_normalization_2")(self.cat_gate)

            # 改进1  自编码器
            # self.encode_input = Average()([self.params_input, self.inputData])
            self.encode_input = GlobalMaxPooling1D()(self.params_input)
            self.encoder_1 = Dense(64, activation='sigmoid')(self.encode_input)
            self.en_dropout_1 = Dropout(0.5)(self.encoder_1)
            self.encoder_2 = Dense(16, activation='sigmoid')(self.en_dropout_1)
            self.en_dropout_2 = Dropout(0.5)(self.encoder_2)
            self.z_c = Dense(1)(self.en_dropout_2)

            self.decoder_1 = Dense(16, activation='sigmoid')(self.z_c)
            self.de_dropout_1 = Dropout(0.5)(self.decoder_1)
            self.decoder_2 = Dense(64, activation='sigmoid')(self.de_dropout_1)
            self.de_dropout_2 = Dropout(0.5)(self.decoder_2)
            self.recon = Dense(102, activation='sigmoid')(self.decoder_2)

            self.dist = Subtract()([self.encode_input, self.recon])

            self.x_lstm = Bidirectional(LSTM(100, return_sequences=True))(self.inputData)
            self.x_lstm_1D = GlobalMaxPooling1D(name="global_max_pooling1d")(self.x_lstm)

            self.dense_1 = Concatenate()([self.x_lstm_1D, self.dist])
            # self.dense_1 = self.x_lstm_1D
            self.dense_2 = Dense(128, activation='relu')(self.dense_1)
            self.dropout_1 = Dropout(0.5)(self.dense_2)
            self.dense_3 = Dense(64, activation='relu')(self.dropout_1)
            self.dropout_2 = Dropout(0.5)(self.dense_3)

            self.fea = Dense(1)(self.dropout_2)

            self.net_output = Activation('sigmoid')(self.fea)

            model = keras.models.Model(inputs=[self.params_input], outputs=self.net_output)
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        else:
            model = load_model(model_path)

        model.summary()

        # keras.utils.vis_utils.plot_model(model, to_file='./struct_model/model_init.png',
        #                                  show_shapes=True)
        return model

    def train(self, max_epoch, batch_size, x_train, y_train, x_val, y_val, x_test, y_test):
        model = self.get_model()
        class_name = self.__class__.__name__

        print('Length of the train: ', len(x_train))
        print('Length of the validation: ', len(x_val))
        print('Length of the test: ', len(x_test))

        training_generator = ClassifyGenerator(range(len(x_train)), x_train, y_train, batch_size)
        validation_generator = ClassifyGenerator(range(len(x_val)), x_val, y_val, batch_size, shuffle=False)

        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
        callbacks_list = [es]


        # If the program is running on Windows OS, you can remove "use_multiprocessing=True," and "workers=6,".
        model.fit_generator(generator=training_generator,
                            validation_data=validation_generator,
                            epochs=max_epoch,
                            callbacks=callbacks_list
                            )
        return model

def predict(model_name, data, label):
    model = load_model(model_name)
    validation_generator = ClassifyGenerator(range(len(data)), data, label, 10, shuffle=False)
    y_pred = model.predict_generator(generator=validation_generator, max_queue_size=10, verbose=1)
    return y_pred

#后续仔细看代码
def plot_recall(y_true, y_pred):
    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx

    fpr, tpr, threshold = roc_curve(y_true[:len(y_pred)], y_pred)
    roc_auc = auc(fpr, tpr)

    tmp_df = pd.DataFrame({'fpr': fpr, 'tpr': tpr}).groupby('fpr').max()
    tpr = tmp_df['tpr'].ravel()
    fpr = tmp_df.index.ravel()

    idx = find_nearest(fpr, 0.001)
    print("fpr", fpr[idx])

    plt.figure()
    lw = 2
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='#fb8072',
             lw=lw, label='AUC=%0.5f' % roc_auc, linestyle='-')

    plt.xlim([0.0, 0.01])
    plt.ylim([0.0, 1.05])
    plt.xlabel('FPR')
    plt.ylabel('Recall')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.vlines(fpr[idx], 0, tpr[idx], colors='#b3de69', linestyles="dashed")
    plt.hlines(tpr[idx], 0, fpr[idx], colors='#b3de69', linestyles="dashed")
    plt.annotate(r'$recall={:.5f}$'.format(tpr[idx]), xy=(0.0004, tpr[idx]), xycoords='data', xytext=(-10, +20),
                 textcoords='offset points', fontsize=12)
    plt.annotate(r'$fpr=%.4f$' % fpr[idx], xy=(0.0004, 0.4), xycoords='data', xytext=(+30, -30),
                 textcoords='offset points', fontsize=12)
    print("tpr", tpr[idx])
    print("auc", roc_auc)

    plt.show()

if __name__=="__main__":
    start_time = time.time()
    #准备数据
    labels = pd.read_csv("../Dataset/label_15000.csv")
    X_May = []
    X_Apr = []
    for row in labels.iterrows():
        file_name = row[1]['file_name']
        label = row[1]['is_malicious']
        if file_name.startswith("201704"):
            X_Apr.append({"file_name": file_name, "label": label})
        else:
            X_May.append({"file_name": file_name, "label": label})
    X_Apr = pd.DataFrame(X_Apr)
    X_May = pd.DataFrame(X_May)

    #交叉验证
    n_fold = 2
    folds = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=0)
    #这里采用四月的数据训练，五月的数据测试 后续可以使用四月数据测试进行对比
    y = X_Apr.label.ravel()
    X = X_Apr.drop(columns=['label'])
    y_test = X_May.label.ravel()
    x_test = X_May.drop(columns=['label'])

    # model_name = "Mal-CLAM.h5"
    # y_pred = predict(model_name, x_test, y_test)
    # print(y_pred[0])
    # plot_recall(y_test, y_pred)


    model_name = "Mal-CLAM.h5"
    MalCLAM = load_model(model_name)
    dim = 1000
    #201705_0 (51,102)
    base_path = "D:/test/{0}.npy"
    npy_name = "8FFCD330EDAEFAC6320E9D00D39EDBE2D396BCC5FE4D55EFC1D5082B78E9500D_180252_1"
    sample = np.load(base_path.format(npy_name))
    # sample = np.expand_dims(sample,axis=0)
    sample = np.clip(sample, -100, 100)
    predict_sample = np.zeros((1, dim, 102), dtype=float)
    if sample.shape[0] > dim:
        predict_sample[0] = sample[:dim, :]
    else:
        predict_sample[0, :sample.shape[0], :] = sample[:, :]

    res = MalCLAM.predict(predict_sample)
    if(res > 0.5):
        print("该软件是恶意软件")
    else:
        print("该软件不是恶意软件")


    #计算结束时间
    end_time = time.time()
    print("time:", (end_time - start_time) / 3600)