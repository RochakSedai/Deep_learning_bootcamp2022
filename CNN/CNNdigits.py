import matplotlib.pyplot as plt 
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.normalization.batch_normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

# we are going to download minst data set
# 60,000 training sample and 10,000 images in the test set
(X_train, y_train),  (X_test, y_test) = mnist.load_data()
# print(X_train.shape)
# print(y_train.shape)
# print(X_test.shape)
# print(y_test.shape)

# lets plot a grayscale image with the label
# plt.imshow(X_train[0], cmap='gray')
# plt.title('class' +str(y_train[0]))
# plt.show()

# tensorflow can handle format: (batch, height, width, channel)
features_train =  X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
features_test =  X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')

# very similar to min-max normalization: we transform the values
# within the range [0,1] usual
features_train /= 255.0
features_test /= 255.0

# we have 10 output classes we want to end up with one hot
# i.e. 0 --> (1 0 0 0 0 0 0 0 0 0),  9 --> (0 0 0 0 0 0 0 0 0 1)
targets_train = np_utils.to_categorical(y_train, 10)
targets_test = np_utils.to_categorical(y_test, 10)


# lets build the convolution neural network model
model = Sequential()

# input is a  28x28 pixels image
# 32 is a number of filters - (3,3) size of the filter  i.e 32--> kernel or filter
model.add(Conv2D(32, (3,3), input_shape=(28, 28, 1)))
model.add(Activation('relu'))

# normalizes the activation in the previouis layer after the convolution phase
# transformation maintains the mean activation close to 0 std(standard deviation) close to 1

model.add(BatchNormalization())
model.add(Conv2D(32, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(BatchNormalization())

# regularization helps to avoid overfitting
model.add(Dropout(0.3))
model.add(Dense(10, activation='softmax'))

#model.summary()
model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

# model.fit(features_train, targets_train, batch_size=128, epochs=5, validation_data=(features_test, targets_test), verbose=1)

# score = model.evaluate(features_test, targets_test)
# print('Test accuracy: %.2f' % score[1])

# data augmnetation helps to reduce overfitting
train_generator =  ImageDataGenerator(rotation_range=7, width_shift_range=0.05, shear_range=0.04, height_shift_range=0.07, zoom_range=0.05)
test_generator = ImageDataGenerator()

train_generator = train_generator.flow(features_train, targets_train, batch_size=64)
test_generator = test_generator.flow(features_test, targets_test, batch_size=64)

model.fit_generator(train_generator, steps_per_epoch=60000//64, epochs=5, validation_data=test_generator, validation_steps=10000//64)
