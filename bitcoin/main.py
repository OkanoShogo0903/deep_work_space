import sys
import math
import types
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.python import debug as tf_debug
from tensorflow.python.debug.lib.debug_data import has_inf_or_nan

import keras
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.utils import plot_model
from keras import backend as K

# データセットの読み込み  
# Timestamp,Open,High,Low,Close,Volume_(BTC),Volume_(Currency),Weighted_Price
df = pd.read_csv('./coinbaseUSD_1-min_data_2014-12-01_to_2018-11-11.csv', sep=',')
df.columns = ['time', 'open', 'high', 'low', 'close', 'volume_btc', 'volume_currency', 'weight_price']
df = df.fillna(df.mean())

def set_debugger_session():
    sess = K.get_session()
    sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    sess.add_tensor_filter('has_inf_or_nan', has_inf_or_nan)
    K.set_session(sess)

def make_dataset(low_data, n_prev=100):

    data, target = [], []
    maxlen = 25

    for i in range(len(low_data)-maxlen):
        data.append(low_data[i:i + maxlen])
        target.append(low_data[i + maxlen])

    re_data = np.array(data).reshape(len(data), maxlen, 1)
    re_target = np.array(target).reshape(len(data), 1)

    return re_data, re_target


if __name__ == '__main__':
    is_load = None
    if len(sys.argv) > 1:
        for argv in sys.argv[1:]: 
            if argv == '--debug':
                set_debugger_session()
            elif argv == '--load':
                is_load = True
            else:
                raise ValueError('unkown option :{}'.format(sys.argv[1]))

    #x_train -> 学習データ，y_train -> 学習ラベル
    LABEL = 'open'
    DATASET_LATE = 0.8
    train_data = df[LABEL].values[:math.floor(len(df)*DATASET_LATE)-1]
    test_data  = df[LABEL].values[math.floor(len(df)*DATASET_LATE):]
    #train_data = df[LABEL].values[2014000:2016000]
    #test_data  = df[LABEL].values[2016001:]
    x_train, y_train = make_dataset(train_data)
    x_test , y_test  = make_dataset(test_data)

    # モデル構築
    # 1つの学習データのStep数(今回は25)
    length_of_sequence = x_train.shape[1] 
    in_out_neurons = 1
    n_hidden = 300

    model = Sequential()
    model.add(LSTM(n_hidden, batch_input_shape=(None, length_of_sequence, in_out_neurons), return_sequences=False))
    model.add(Dense(in_out_neurons))
    model.add(Activation("linear"))
    optimizer = Adam(lr=1.0)
    model.compile(loss="mean_squared_error", optimizer=optimizer)

    # Learning
    early_stopping = EarlyStopping(monitor='val_loss', mode='auto', patience=20)
    if is_load != True:
        model.fit(x_train, y_train,
                batch_size=300,
                epochs=1,
                validation_split=0.1,
                callbacks=[early_stopping]
                )

        # Save model image 
        plot_model(model, to_file='model.png')
        # Model save
        model.save("bitcoin.h5")

    # Model load
    load_model = load_model("bitcoin.h5")

    # Predict
    print ("x_test")
    print (x_test)
    print ("y_test")
    print (y_test)
    predicted = load_model.predict(x_test)

    # Plot
    print (predicted)
    plt.figure()
    plt.plot(range(25,len(predicted)+25), predicted, color="r", label="predict_data")
    plt.plot(range(0, len(y_test)), y_test, color="b", label="raw_data")
    plt.legend()
    plt.show()

