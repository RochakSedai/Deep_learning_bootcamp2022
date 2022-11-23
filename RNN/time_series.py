import numpy as np 
import matplotlib.pyplot as plt 
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense 
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.layers import Dropout

NUM_OF_PREV_ITEMS = 5

def reconstruct_data(data_set, n=1):
    x, y = [], []

    for i in range(len(data_set)-n-1):
        a = data_set[i:(i+n), 0]
        x.append(a)
        y.append(data_set[i+n, 0])

    return np.array(x), np.array(y)



# WE wawnt to make sure the results will be hte same
# everytime we run the algo.
np.random.seed(1)

#load the dataset
data_frame = read_csv('daily_min_temperatures.csv', usecols=[1])

# print(data_frame)
# print(data_frame.values)

# we just need  temperature column
data = data_frame.values
# we are dealing with floating point values
data = data.astype('float32')

# min-max normalization
scaler = MinMaxScaler(feature_range=(0,1))
data = scaler.fit_transform(data)
# print(data)

# split the train and test sets ( 70-30)
train, test = data[0:int(len(data) * 0.7), :], data[int(len(data) * 0.7):len(data), :]

# create the training data and test data matrix
train_x, train_y = reconstruct_data(train, NUM_OF_PREV_ITEMS)
test_x, test_y = reconstruct_data(test, NUM_OF_PREV_ITEMS)

# print(train_x)
# print(train_y)


 # reshape input to be [numOfSamples, time steps, numOfFeatures]
 # time step is 1 becaue we want to predict the next value (t+1)

print((train_x.shape[0], 1, train_x.shape[1]))
train_x = np.reshape(train_x, (train_x.shape[0], 1, train_x.shape[1]))
test_x = np.reshape(test_x, (test_x.shape[0], 1, test_x.shape[1]))


# create the LSTM model
model = Sequential()
model.add(LSTM(units=100, return_sequences=True, input_shape=(1, NUM_OF_PREV_ITEMS)))
model.add(Dropout(0.5))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(units=50))
model.add(Dropout(0.3))
model.add(Dense(units=1))

# optimize the model with ADAM optmizer
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(train_x, train_y, epochs=10, batch_size=16, verbose=2)

# make predictions and min-max normalization
test_predict = model.predict(test_x)
test_predict = scaler.inverse_transform(test_predict)
test_labels = scaler.inverse_transform([test_y])

test_score = mean_squared_error(test_labels[0], test_predict[:, 0])
print('Score on the test set: %.2f MSE' % test_score)

# PLOT THE RESULT (ORIGINAL DATA + POREDICTION)
test_predict_plot = np.empty_like(data)
test_predict_plot[:, :] = np.nan
test_predict_plot[len(train_x)+2*NUM_OF_PREV_ITEMS+1:len(data)-1, :] = test_predict
print(test_predict_plot)

plt.plot(scaler.inverse_transform(data))
plt.plot(test_predict_plot, color='green')
plt.show()