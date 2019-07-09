from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from Image import ImageData
from keras import backend as k
from CNN import Cnn



imdt = ImageData('../PP_Data/images')
cnn = Cnn()
model = cnn.model()
model.fit(imdt.train_x, imdt.train_y, epochs=10, batch_size = 200)
scores = model.evaluate(imdt.test_x, imdt.test_y, verbose=2)
print("la precision sur la base de test est : %.2f%%" % (scores[1] * 100))
