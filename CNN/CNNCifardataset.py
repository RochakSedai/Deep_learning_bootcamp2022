from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.optimizers import SGD
from keras.layers.normalization.batch_normalization import BatchNormalization
import ssl


ssl._create_default_https_context = ssl._create_unverified_context

# load- data =  50K training samples and 10K test samples
# 32x32 pixel image - 10 output classes(labels)
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

print(X_train.shape)

# we have 10 output classes we want to end up with one hot
# i.e. 0 --> (1 0 0 0 0 0 0 0 0 0),  1 --> (0 1 0 0 0 0 0 0 0 0)
y_train =  to_categorical(y_train)
y_test = to_categorical(y_test)

# normalizing the dataset
X_train = X_train/255.0
X_test = X_test/255.0

# cosntruct CNN model
model = Sequential()
model.add(Conv2D(32, (3,3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3))) #32 -> image_height, 32-> image width, 2-> RGB
model.add(BatchNormalization())
model.add(Conv2D(32, (3,3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3,3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3,3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.2))
model.add(Conv2D(128, (3,3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3,3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(10, activation='softmax'))

# training model ; SGD --> Sophastic Gradient Descent
optimizer = SGD(learning_rate=0.001, momentum=0.95)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test), verbose=2)

#evaluate the model
model_result = model.evaluate(X_test, y_test, verbose=0)
print('Accuracy of CNN Model: %s' % (model_result[1] * 100.0))

