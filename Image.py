import os
import shutil
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical


class ImageData:

    def __init__(self,image_dir, train_dir = None, test_dir = None):
        self.image_dir = image_dir
        if train_dir == None:
            self.train_dir = "../PP_Data/train_dir"
        else:
            self.train_dir = train_dir
            
        if test_dir == None:
            self.test_dir = "../PP_Data/test_dir"
        else:
            self.test_dir = test_dir
        # to be excuted only once to create the testing and training directories
        #self.copy_data()
        self.train_x, self.train_y = self.prepare_data(self.train_dir)
        self.test_x, self.test_y = self.prepare_data(self.test_dir)
        

    def prepare_data(self,data_dir):
        data_x,data_y = ([],[])
        labels = []
        for img in os.listdir(data_dir): 
            labels.append(self.get_label(img))
            img = Image.open(data_dir + '/' + img)
            img = img.resize((75, 75), Image.ANTIALIAS)
            img = img.convert('L')
            #print(np.array(img).shape)
            data_x.append(np.array(img))
        data_x = np.stack(data_x)
        data_x = data_x.reshape(data_x.shape[0],75,75,1).astype('float32')
        data_x /=255
        data_y = self.one_hot_encode(labels)
        return data_x,data_y
                    
            

    def one_hot_encode(self,labels):
        values = np.array(labels)
        # encode labels into integers
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(values)
        # use Keras one-hot encoding function
        return to_categorical(integer_encoded)
             

    # select only 4 types of power plant images and split them into train and test data in seperate directories  
    def copy_data(self, args = ['NG','SUN']):
        # Create target Directories if they don't exist
        if not os.path.exists(self.train_dir):
            os.mkdir(self.train_dir)
        if not os.path.exists(self.test_dir):
            os.mkdir(self.test_dir)
            
        for img in os.listdir(self.image_dir):
            label = self.get_label(img)
            if label in args:
                rand = self.bernoulli_dist()
                if rand == 1:
                    shutil.copy2(self.image_dir + '/' + img, self.train_dir)
                    print(img + ' added to train_data')
                elif rand == 0:
                    shutil.copy2(self.image_dir + '/' + img, self.test_dir)
                    print(img + ' added to test_data')

            
    #  image name format : [source]_[id]_[state]_[type of fuel].tif          
    def get_label(self,img_name):
        label = img_name.split('_')[3]
        return label[:label.find('.')]

    # to split images into ~ 80% train data & 20% test data on the fly    
    @staticmethod     
    def bernoulli_dist():
        return np.random.binomial(size = 1, n = 1, p = 0.8)


#imdt.copy_data()

