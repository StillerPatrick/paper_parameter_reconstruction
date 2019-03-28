from os.path import join
import numpy as np
import scipy.misc
import h5py
from Enumerations import DataSets
from tqdm import tqdm




def minmax_norm(x,new_max=1):
    min = x.min()
    max = x.max()
    return (x-min)/(max-min)*new_max

class ScatterImageSet:
    def __init__(self,
                 x_path,
                 y_path,
                 batch_size,
                 training_size,
                 test_size=0,
                 validation_size=0,
                 normalize=False,
                 shift=False):
        '''
        Function initializes the scatter images set. The constructor fixes the size of the training, test and validation
        set. You are able to map the labels into range of [0,1] if you enable the pich normalization flag.
        If you want a shuffled data set activate the shuffle flag.

        :param x_path: the absolute path to the image data set
        :param y_path: the absolute path to the label file
        :param training_size: the number of images in the training_set
        :param test_size: number of images in the test set
        :param validation_size: number of images in the validation set
        :param normalize: Activates scale to [0,1]
        :return returns a Scatter Image Set
        '''


        # INIT
        self.batch_size = batch_size
        self.training_size = training_size
        self.test_size = test_size
        self.validation_size = validation_size
        self.y = np.load(y_path)

        com_img_size = self.y.shape[0]

        self.transform_labels()

        self.x = [join(x_path, '{:06d}_endproduct.h5'.format(i)) for i in range(com_img_size)]

        perm = np.random.RandomState(seed=42).permutation(self.y.shape[0])
        self.x = np.take(self.x, perm, axis=0)
        self.y = np.take(self.y, perm, axis=0)

        # CUT
        if test_size + validation_size + training_size > com_img_size:
            raise ValueError("The sum of test, training size and validation size must be lower equal then ",
                             com_img_size)

        self.training_x = self.x[:training_size]
        self.training_y = self.y[:training_size]

        # Cut out the training_set
        self.x = self.x[training_size:]
        self.y = self.y[training_size:]

        self.validation_x = self.x[:validation_size]
        self.validation_y = self.y[:validation_size]

        # Cut out the validation_set‚
        self.x = self.x[validation_size:]
        self.y = self.y[validation_size:]

        # Take the rest as test set
        self.test_x = self.x[:test_size]
        self.test_y = self.y[:test_size]

        # load data sets

        # training

        result_training = []
        for path in tqdm(self.training_x, desc='Loading Training Set'):
            with h5py.File(path, 'r') as hf:
                result_training.append(hf['endproduct'][:])
        self.training_x = np.array(result_training)

        # test
        result_test = []
        for path in tqdm(self.test_x ,desc='Loading Test Set'):
            with h5py.File(path, 'r') as hf:
                result_test.append(hf['endproduct'][:])
        self.test_x = np.array(result_test)

        # validation
        result_validation = []
        for path in tqdm(self.validation_x, desc='Loading Validation Set'):
            with h5py.File(path, 'r') as hf:
                result_validation.append(hf['endproduct'][:])
        self.validation_x = np.array(result_validation)

        if normalize:
            self.training_x = np.array([minmax_norm(e) for e in self.training_x])
            self.validation_x = np.array([minmax_norm(e) for e in self.validation_x])
            self.test_x = np.array([minmax_norm(e) for e in self.test_x])
            
        if shift:
            self.training_x = np.array([np.fft.fftshift(e) for e in self.training_x])
            self.validation_x = np.array([np.fft.fftshift(e) for e in self.validation_x])
            self.test_x = np.array([np.fft.fftshift(e) for e in self.test_x])





    def transform_labels(self):
        '''
        Function that change labels in form of sigma, pitch and fsize
        :param self:
        :param pitch_normalization: transform labels into scope 0 to 1
        :return:
        '''
        self.y = np.array([[e["fsize"],e["sigma"]] for e in self.y])

    def get_batch(self, idx, data_set=DataSets.TRAINING):
        '''
        :param self:
        :param idx: the iteration in the epoch
        :param data_set: define the active dataset with a enumeration
        :return: return the labeled data in a batch
        '''
        set_x = self.training_x
        set_y = self.training_y

        if data_set == DataSets.VALIDATION:
            set_x = self.validation_x
            set_y = self.validation_y

        if data_set == DataSets.TEST:
            set_x = self.test_x
            set_y = self.test_y

        batch_x = set_x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = set_y[idx * self.batch_size:(idx + 1) * self.batch_size]

        return np.array(batch_x), batch_y




