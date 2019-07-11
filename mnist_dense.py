import numpy as np

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

def reshape_X(x):
    X = x / 255
    X = np.reshape(X, (X.shape[0], 784))
    return X

def reshape_Y(y):
    Y = np.zeros((y.shape[0], 10))
    for i in range(y.shape[0]):
        Y[i, y[i]] = 1

    return Y


(x_train, y_train), (x_test, y_test) = mnist.load_data()

print("Preprocessing training data.. ", end='')
X = reshape_X(x_train)
Y = reshape_Y(y_train)
print("Done\n")

print("Preprocessing eval data", end='')
X_eval = reshape_X(x_test)
Y_eval = reshape_Y(y_test)
print("Done\n")

print("Building model..", end='')
model = Sequential()
model.add(Dense(500, input_dim=784, activation='relu'))
model.add(Dense(250, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='softmax'))
print("Done\n")

print("Defining training procedure..", end='')
opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
print("Done\n")

model.fit(X, Y, epochs=32, batch_size=64, validation_data=(X_eval, Y_eval))

_, accuracy = model.evaluate(X_eval, Y_eval)
print('Accuracy: %.2f' % (accuracy*100))

u_input = input("Save? Y/N")
if u_input == 'Y':
    model.save('saved_model.h5')
