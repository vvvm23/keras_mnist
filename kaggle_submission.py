import numpy as np
from keras.models import load_model
from numpy import loadtxt

model = load_model('saved_model.h5')
testset = loadtxt("test.csv", delimiter=',')
testset = testset / 255
f = open("kaggle_submission.csv", 'w')

f.write("ImageId,Label\n")
for i in range(testset.shape[0]):
    probs = model.predict(testset[i, :].reshape((1, -1)))
    max_index = np.argmax(probs)
    f.write("{0},{1}\n".format(i+1, max_index))
f.close()