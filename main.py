from keras.optimizers import Adam
from Image import ImageData
from CNN import Cnn
import keras


batch_size = 128
num_classes = 2
epochs = 15


imdt = ImageData('../PP_Data/images')
cnn = Cnn()
model = cnn.model3()
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(imdt.train_x, imdt.train_y,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(imdt.test_x, imdt.test_y))
score = model.evaluate(imdt.test_x, imdt.test_y, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


