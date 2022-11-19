from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam 
import pandas as pd
import numpy as np

creditData = pd.read_csv('credit_data.csv')

features = creditData[['income', 'age', 'loan']]

y = np.array(creditData.default).reshape(-1,1)

encoder  = OneHotEncoder()
targets = encoder.fit_transform(y).toarray()

feature_train, feature_test, target_train, target_test = train_test_split(features, targets, test_size=0.2)

model = Sequential()
model.add(Dense(10, input_dim=3, activation='sigmoid'))
model.add(Dense(2, input_dim=10, activation='softmax'))

optimizer = Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy',
                optimizer=optimizer,
                metrics=['accuracy'])

model.fit(feature_train, target_train, epochs=1000, verbose=2)
results = model.evaluate(feature_test, target_test, use_multiprocessing=True)


print('Training is finished .... The loss and accuracy vlaues are: ')
print(results)




