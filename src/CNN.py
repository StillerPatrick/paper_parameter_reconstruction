import tensorflow as tf
import numpy as np
from DataSetSigmaFsizePitch import ScatterImageSet
from Enumerations import DataSets
from tqdm import tqdm
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class PITCH_CNN:
    def __init__(self,
                 save_dir,
                 tensorboard_path,
                 num_conv_layer,
                 filter_size,
                 num_filters,
                 residual_distance,
                 num_dense_units,
                 learning_rate):
        """
        :param savedir: The savepath of tensorflow checkpoints
        :param tensorboardpath: The path for tensorflow logging
        :param num_conv_layer: defines the number of convolutional layers added to the graph
        :param filter_size: defines the size of the filters in the convolutional layers
        :param num_filters: defines the number of convolutional layers added to convolutional layer
        :param residual_distance: defines the distance and the stepsize of residual layers
         """
        # set paramter for saving the checkpoints
        self.save_dir = save_dir
        # set parameter for logging the the training process
        self.tensorboard_path = tensorboard_path
        # building the graph with hyperparameters

        self.build(num_conv_layer,
                   filter_size,
                   num_filters,
                   residual_distance,
                   num_dense_units,
                   learning_rate)

        self.saver = tf.train.Saver()

    def costumized_cnn_layer(self,input,filter_size,filters,name):
        with tf.name_scope(name):
            cnn = tf.layers.conv1d(inputs=input,kernel_size=filter_size,filters=filters,padding='same')
            output = tf.nn.relu(tf.layers.batch_normalization(inputs=cnn,training=self.train))
            return output



    def build(self,
              num_conv_layer,
              filter_size,
              num_filters,
              residual_distance,
              learning_rate):
        """
        :param num_conv_layer: defines the number of convolutional layers added to the graph
        :param filter_size: defines the size of the filters in the convolutional layers
        :param num_filters: defines the number of convolutional layers added to convolutional layer
        :param residual_distance: defines the distance and the stepsize of residual layers
        :return: returns the loss of the model adds the predict operation to the model
        """
        print("Hyperparameters")
        #hyperparams = locals()
        #print(hyperparams)
        print("Num Conv",num_conv_layer)
        print("Filter Size",filter_size)
        print("Num Filters",num_filters)
        print("Residual Distance",residual_distance)
        print("Num Dense",num_dense_units)
        print("Learning Rate",learning_rate)
        
        tf.reset_default_graph()
        # Build Placeholder for netinput and layer
        self.x = tf.placeholder(tf.float64, [None, 2048], name="x")
        self.y = tf.placeholder(tf.float64, name="y")
        self.train = tf.placeholder(tf.bool,name="train")
        #self.batch_size = tf.placeholder(tf.int,name="batch_size")

        features = tf.reshape(self.x,(-1,2048,1))

        # array that holds all convolutional layer
        conv_layer = []

        # add fist layer that gets input from x

        conv_layer.append(tf.layers.conv1d(inputs=features,
                                           kernel_size=filter_size,
                                           filters=num_filters,
                                           padding = 'same',
                                           name="ConvLayer0"))

        # create Deep Convolutional Neuronal Layer with Ressidual Steps
        for i in range(1, num_conv_layer):
            # every N Steps add one residual layer
            if i % residual_distance == 0:
                conv_layer.append(tf.layers.conv1d(inputs=tf.add(conv_layer[i - 1],conv_layer[i - residual_distance]),
                                                   kernel_size=filter_size,
                                                   filters=num_filters,
                                                   padding = 'same',
                                                   name="ConvLayer" + str(i)))
            else:
                # else add normal convolutional layer
                conv_layer.append(tf.layers.conv1d(inputs=conv_layer[i - 1],kernel_size=filter_size,filters=num_filters,padding='same',name="ConvLayer" + str(i)))
        # now summing up for the flatten layer
        last_conv = tf.layers.conv1d(inputs=conv_layer[-1],kernel_size=1,filters=1,padding='same',name="DownSampling")
        flatten = tf.layers.flatten(last_conv,name="Flatten")
        print("Flatten Shape:",flatten.get_shape())
        dense = tf.layers.dense(inputs=flatten,units=60,activation=None,name="Dense")
        self.predict_op = tf.layers.dense(inputs=dense, units=1, activation=None, name="Output")

        # Add Summary for loss to Graph
        self.loss = tf.losses.mean_squared_error(self.y,self.predict_op)
        tf.summary.scalar('Loss', self.loss)

        global_step = tf.Variable(0, trainable=False)
      
        self.train_op =  tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss,global_step=global_step)

    def fit(self, epochs, batch_size, dataset,restore=""):
        """
        Function that trains the neuronal network
        :param epochs: how many times iterate over the dataset
        :param batch_size: the size of the data that uses for fitting
        :param dataset: tha database
        :return:
        """
      
        sess = tf.Session()

        # histograms for variables
        for var in tf.trainable_variables():
            tf.summary.histogram(var.name, var)
        
      
        merged = tf.summary.merge_all()
        sess.run(tf.global_variables_initializer())
        
        if restore != "":
            self.saver.restore(sess,restore)

        # Add ops to save and restore all the variables.

        training_writer = tf.summary.FileWriter(self.tensorboard_path + '/training',
                                                sess.graph)
        test_writer = tf.summary.FileWriter(self.tensorboard_path+'/test')
        
        num_training_batches = dataset.training_size // batch_size
        num_test_batches = dataset.test_size // batch_size
        
        best_val = np.inf
        best_training = np.inf
        
        rand_state = np.random.RandomState(seed=42)
        batch_idx = np.arange(num_training_batches)
       
        for epoch in range(epochs):
            batch_idx = rand_state.permutation(batch_idx)
            sys.stdout.flush()# Epoch Loop runs HM_EPOCHS TIMES
            epoch_loss = 0  # initialize the epoch loss, this value will added up over all batches
            for idx in batch_idx:
                # LEARNING STEP
                epoch_x, epoch_y = dataset.get_batch(idx, data_set=DataSets.TRAINING)
                # current batch over the training
                _, c, prediction = sess.run([self.train_op, self.loss,self.predict_op], feed_dict={self.x: epoch_x, self.y: epoch_y,self.train: True})
                epoch_loss += c  # add batch loss too epoch loss
            
    
            print("Loss at Epoch", epoch + 1,"out of", epochs, "is", epoch_loss /num_training_batches)

            
            # DOCUMENTATION OF THE LEARNING SUCCESS

            if (epoch + 1) % 50 == 0:
                # WRITE VALIDATION SUCCESS EVERY 5´th epoch
                summary, c = sess.run([merged, self.loss], feed_dict={self.x: dataset.validation_x, self.y: dataset.validation_y,self.train: False})
                test_writer.add_summary(summary, epoch)
                print('Valitdation Loss at step %s: %s' % (epoch + 1, c))
                if c < best_val :
                    best_val = c
                    #callback save better validation loss
                    print("New Best State")
                    self.saver.save(sess,self.save_dir,global_step=epoch)
                
                # WRITE EVERY 10th EPOCH THE TRAINING SUCCESS
                training_x, training_y = dataset.get_batch(0, data_set=DataSets.TRAINING)
                print(training_y)
                summary, c, prediction = sess.run([merged, self.loss,self.predict_op], feed_dict={self.x: training_x, self.y: training_y,self.train: False})
                print("Predicts:", prediction, " for ", training_y)
                training_writer.add_summary(summary, epoch)
            
        print("======= TEST PERFORMANCE =======")
        # TEST PERFORMANCE
        test_loss = 0 
        predictions = []
        labels = []
        for idx in range(num_test_batches):
            test_x, test_y = dataset.get_batch(idx, data_set=DataSets.TEST)
            labels.extend(test_y)
            loss,prediction = sess.run([self.loss,self.predict_op],feed_dict={self.x : test_x,self.y:test_y[:,0],self.train: False})
            prediction = np.reshape(prediction,(batch_size))
            predictions.extend(prediction)
            test_loss += loss
            for i in range(batch_size):
                print("Label:",test_y[i],"Prediction:",prediction[i])
        #self.saver.save(sess,self.save_dir)
        print("Test_Loss:", test_loss / num_test_batches)
        np.save('cnn_predictions.npy',predictions)
        np.save('cnn_labels.npy',labels)
        sess.close()
        return best_val, best_training

    def predict(self, x):
        """
        :param x: input x
        :return: Returns the prediction of the model for input x
        """
        with tf.Session() as session:
            # restore the model
            #print(self.save_dir)
            self.saver.restore(session,self.save_dir)
            P = session.run(self.predict_op, feed_dict={self.x: x})
        return P
