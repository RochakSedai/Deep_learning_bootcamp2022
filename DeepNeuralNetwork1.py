import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense


# Why XOR? Because it is non_linear problem
X =  np.array([[0,0], [0,1], [1,0], [1,1]], 'float32')
y = np.array([[0], [1], [1], [0]], 'float32')

# we can define neural network model in a sequential manner
model = Sequential()

#first parameter is output dimension
model.add(Dense(16, input_dim=2, activation='relu'))
model.add(Dense(16, input_dim=16, activation='relu'))
model.add(Dense(16, input_dim=16, activation='relu'))
model.add(Dense(16, input_dim=16, activation='relu'))
model.add(Dense(16, input_dim=16, activation='relu'))
model.add(Dense(16, input_dim=16, activation='relu'))
model.add(Dense(16, input_dim=16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# we can define the loss function MSE or negative log likelihood
# optimizer will find the right adjustment of weight
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['binary_accuracy'])

# epoch is a iteration over the entire dataset
# verbose 0 is silent 1 and 2 are showing results
model.fit(X, y, epochs=500, verbose=2)

print('Predictions after the training ...')
print(model.predict(X).round())
