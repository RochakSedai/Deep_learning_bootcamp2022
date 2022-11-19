from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam 


iris_data = load_iris()

features = iris_data.data 
labels = iris_data.target.reshape(-1,1)

# we have 3 classes so the label will have 3 values
# first class: (1,0,0) second class: (0,1,0) third class: (0,0,1)
encoder = OneHotEncoder()
targets = encoder.fit_transform(labels).toarray()

feature_train, feature_test, target_train, target_test = train_test_split(features, targets, test_size=0.2)

model = Sequential()
model.add(Dense(10, input_dim=4, activation='sigmoid'))
model.add(Dense(3, input_dim=10, activation='softmax'))

# we can define the loss function MSE or negative log likelihood
# optimizer will find the right adjustments for the weights: SGD, Adagrad, ADAM ..

optimizer = Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy',
                optimizer=optimizer,
                metrics=['accuracy'])

model.fit(feature_train, target_train, epochs=10000, batch_size=20, verbose=2)
results = model.evaluate(feature_test, target_test, use_multiprocessing=True)

print('Training is finished .... The loss and accuracy vlaues are: ')
print(results)


