import sys
import types
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.python import debug as tf_debug
from tensorflow.python.debug.lib.debug_data import has_inf_or_nan

import keras
from keras.models import Sequential
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
print ("***", type(df))
df = df.fillna(df.mean())
print ("***", type(df))
#print (df)

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
    if len(sys.argv) > 1:
        if sys.argv[1] == '--debug':
            set_debugger_session()
        else:
            raise ValueError('unkown option :{}'.format(sys.argv[1]))

    #g -> 学習データ，h -> 学習ラベル
    f = df['open'].values[0:2014000]
    #f = df['open'].values[2016200:2016300]
    g, h = make_dataset(f)
    #print ("g", "-"*50)
    #print (g)
    #print (g.shape)
    #print ("h", "-"*50)
    #print (h)
    #print (h.shape)

    # モデル構築
    # 1つの学習データのStep数(今回は25)
    length_of_sequence = g.shape[1] 
    in_out_neurons = 1
    n_hidden = 300

    model = Sequential()
    model.add(LSTM(n_hidden, batch_input_shape=(None, length_of_sequence, in_out_neurons), return_sequences=False))
    model.add(Dense(in_out_neurons))
    model.add(Activation("linear"))
    optimizer = Adam(lr=0.05)
    model.compile(loss="mean_squared_error", optimizer=optimizer)

    # Learning
    early_stopping = EarlyStopping(monitor='val_loss', mode='auto', patience=20)
    model.fit(g, h,
            batch_size=300,
            epochs=100,
            validation_split=0.1,
            callbacks=[early_stopping]
            )

    plot_model(model, to_file='model.png')
    # 予測

    predicted = model.predict(g)

    plt.figure()
    plt.plot(range(25,len(predicted)+25), predicted, color="r", label="predict_data")
    plt.plot(range(0, len(f)), f, color="b", label="row_data")
    plt.legend()
    plt.show()

