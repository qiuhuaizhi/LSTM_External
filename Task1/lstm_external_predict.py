

import sys
import codecs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, Activation, merge, Input, Reshape
from keras.optimizers import Adam
def load_external_data(file_name):
    df = pd.read_csv(file_name, sep=',', usecols=[0,1,2,7,8,9,10,18,19,20])
    dayOfWeek = pd.get_dummies(df['dayOfWeek'])
    dayOfYear = pd.get_dummies(df['dayOfYear'])
    dayOfMonth = pd.get_dummies(df['dayOfMonth'])
    df = pd.concat([df, dayOfWeek, dayOfYear, dayOfMonth], axis=1)
    df = df.drop(['dayOfWeek', 'dayOfYear', 'dayOfMonth'], axis=1)
    external_data = np.array(df).astype(float)
	
def load_data(file_name, sequence_length=10, split=0.8):
    df = pd.read_csv(file_name, sep=',', usecols=[0,1,2,3,4,5,6,7,8,9,10,17,18,19,20,27])
    df = df.dropna()
    df = df.reset_index()

    df_timeseries = df['zonalPrice']
    df_external = df.drop(['zonalPrice', 'index'], axis=1)
    print df_external.columns.shape
    data_all = np.array(df_timeseries).astype(float)
    scaler = MinMaxScaler()
    data_all = scaler.fit_transform(data_all)
   
    dayOfWeek = pd.get_dummies(df_external['dayOfWeek'])
    dayOfYear = pd.get_dummies(df_external['dayOfYear'])
    dayOfMonth = pd.get_dummies(df_external['dayOfMonth'])
    weekOfYear = pd.get_dummies(df_external['weekOfYear'])
    hourOfDay = pd.get_dummies(df_external['hourOfDay'])
    monthOfYear = pd.get_dummies(df_external['monthOfYear'])
    
    df_time = pd.concat([dayOfWeek, dayOfYear, dayOfMonth, weekOfYear, hourOfDay, monthOfYear], axis=1)
    
    #df_time = df_external[['dayOfWeek', 'dayOfYear', 'dayOfMonth', 'weekOfYear', 'hourOfDay', 'monthOfYear']]
    df_load = df_external.drop(['dayOfWeek', 'dayOfYear', 'dayOfMonth', 'weekOfYear', 'hourOfDay', 'monthOfYear'], axis=1)
    print df_load.columns
    external_time = np.array(df_time).astype(int)
    print external_time.shape
    external_load = np.array(df_load).astype(float)
    print external_load.shape
    external_all = np.hstack((external_time, external_load))
    #external_all = external_time   
    data = []
    for i in range(len(data_all) - sequence_length):
	feature = np.concatenate([external_all[i+sequence_length], data_all[i: i+sequence_length+1]], axis=0)
        data.append(feature)
    reshaped_data = np.array(data)
    #np.random.shuffle(reshaped_data)
    
    x = reshaped_data[:, :-1]
    #x = scaler.fit_transform(x)
    y = reshaped_data[:, -1]
    split_boundary = -25
    train_x = x[: split_boundary]
    test_x = x[split_boundary:]

    train_y = y[: split_boundary]
    test_y = y[split_boundary:]

    return train_x, train_y, test_x, test_y, scaler


def build_model(external_dim=10):
    main_inputs = []
    # main outputs using lstm
    input1 = Input(shape=(10, ))
    #main_inputs.append(input1)
    reshape = Reshape((10, 1))(input1)
    
    lstm1 = LSTM(output_dim=50, return_sequences=True)(reshape) 
    
    lstm2 = LSTM(100, return_sequences=False)(lstm1)
    
    #dense = Dense(output_dim=1)(lstm2)
    #main_output = Activation('linear')(dense)
    
    #if external_dim != None and external_dim > 0:
    external_input = Input(shape=(external_dim, ))
    main_inputs.append(external_input)
    embedding = Dense(output_dim=20)(external_input)
    embedding = Activation('relu')(embedding)
    h1 = Dense(output_dim=100)(embedding)
    h2 = Activation('relu')(h1)
    #h2 = Dense(output_dim=1)(h1)
    #external_output = Activation('linear')(h2)
    
    main_inputs.append(input1)
    #main_output = merge([main_output, external_output], mode='sum')
    main_output = merge([lstm2, h2], mode='sum')
    main_output = Dense(output_dim=1)(main_output)
    main_output = Activation('linear')(main_output)

    model = Model(input=main_inputs, output=main_output)
    model.summary()
    model.compile(loss='mse', optimizer='adam')
    return model


def train_model(train_x, train_y, test_x, test_y):

    #try:
    train_x = train_x.reshape((train_x.shape[0], train_x.shape[1]))
    model = build_model(train_x.shape[1] - 10)
    model.fit([train_x[:, :-10], train_x[:, -10:]], train_y, batch_size=512, nb_epoch=25, validation_split=0.1)
    test_x = test_x.reshape((test_x.shape[0], test_x.shape[1]))
    predict = model.predict([test_x[:, :-10], test_x[:, -10:]])
    predict = np.reshape(predict, (predict.size, ))
    #except KeyboardInterrupt:
        #print(predict)
        #print(test_y)
    #print(predict)
    #print(test_y)
    #try:
        #fig = plt.figure(1)
        #plt.plot(predict, 'r:')
        #plt.plot(test_y, 'g-')
        #plt.legend(['predict', 'true'])
    #except Exception as e:
    #    print(e)
    return predict, test_y


if __name__ == '__main__':
    taskName = sys.argv[1]
    train_x, train_y, test_x, test_y, scaler = load_data(taskName + 'readyDataMinus.csv')
    train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1))
    test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], 1))
    predict_y, test_y = train_model(train_x, train_y, test_x, test_y)
    
    print predict_y
    print test_y
    predict_y = scaler.inverse_transform([[i] for i in predict_y])
    test_y = scaler.inverse_transform(test_y)

    #print predict_y.reshape((1, 24))[0]
    
    fl = open(taskName + 'ExternalTimeResult.csv', 'w')
    fl.write(str(predict_y.reshape((1, 25))[0]))
    fl.close()
    print taskName + 'ExternalTimeResult.txt'
    #fig2 = plt.figure(2)
    #plt.plot(predict_y, 'g:')
    #plt.plot(test_y, 'r-')
    #plt.show()

