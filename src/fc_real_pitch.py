import tensorflow as tf
import numpy as np
import os
from DataSetSigmaFsizePitch import ScatterImageSet
from Enumerations import DataSets
from CNN import PITCH_CNN
from tqdm import tqdm
from dirac import create_dirac
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'




class FC:
    def __init__(self,
                 save_dir,
                 tensorboard_path,
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

        self.build(learning_rate)
        
        self.saver = tf.train.Saver()




    def build(self,learning_rate):
        """
        :param num_conv_layer: defines the number of convolutional layers added to the graph
        :param filter_size: defines the size of the filters in the convolutional layers
        :param num_filters: defines the number of convolutional layers added to convolutional layer
        :param residual_distance: defines the distance and the stepsize of residual layers
        :return: returns the loss of the model adds the predict operation to the model
        """
    
        tf.reset_default_graph()
        # Build Placeholder for netinput and layer
        self.x = tf.placeholder(tf.float64, [None, 2048], name="x")
        self.dirac = tf.placeholder(tf.float64,[None,2048],name="dirac")
        self.y = tf.placeholder(tf.float64,name="y")
        self.train = tf.placeholder(tf.bool,name="train")
        #self.batch_size = tf.placeholder(tf.int,name="batch_size")

        features = tf.concat([self.x,self.dirac],1)
        
        fc1 = tf.layers.dense(inputs=self.x,units=2048,activation=tf.nn.relu,name="FC1")
        fc2 = tf.layers.dense(inputs=fc1,units=2048,activation=tf.nn.relu,name="FC2")
        fc3 = tf.layers.dense(inputs=fc2,units=2048,activation=tf.nn.relu,name="FC3")
        fc4 = tf.layers.dense(inputs=fc3,units=2048,activation=tf.nn.relu,name="FC4")
        fc5 = tf.layers.dense(inputs=fc4,units=2048,activation=tf.nn.relu,name="FC5")
        fc6 = tf.layers.dense(inputs=fc5,units=2048,activation=tf.nn.relu,name="FC6")
        fc7 = tf.layers.dense(inputs=fc6,units=64,activation=tf.nn.relu,name="FC7")
        fc8 = tf.layers.dense(inputs=fc7,units=64,activation=tf.nn.relu,name="FC8")
        self.predict_op = tf.layers.dense(inputs=fc8, units=1, activation=None, name="Output")

        # Add Summary for loss to Graph
        self.loss = tf.losses.mean_squared_error(self.y,self.predict_op)
        tf.summary.scalar('Loss', self.loss)

        optimizer =  tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.grads = optimizer.compute_gradients(self.loss,tf.trainable_variables())
        self.train_op = optimizer.apply_gradients(self.grads)

    def fit(self, epochs, batch_size, dataset,model):
        """
        Function that trains the neuronal network
        :param epochs: how many times iterate over the dataset
        :param batch_size: the size of the data that uses for fitting
        :param dataset: tha database
        :param model: path to the pitch model 
        :return:
        """
        sess = tf.Session()

        # histograms for variables
        for var in tf.trainable_variables():
            tf.summary.histogram(var.name, var)
        for grad, var in self.grads:
            tf.summary.histogram(var.name + '/gradient', grad)
        
        print(tf.trainable_variables())
      
        merged = tf.summary.merge_all()
        sess.run(tf.global_variables_initializer())

        #pitch_model = PITCH_CNN(model,"",num_conv_layer=10,filter_size=16,num_filters=32,residual_distance=2,num_dense_units=40,learning_rate=1e-8)
                 
        # Add ops to save and restore all the variables.

        training_writer = tf.summary.FileWriter(self.tensorboard_path + '/training',
                                                sess.graph)
        test_writer = tf.summary.FileWriter(self.tensorboard_path+'/test')
        
        num_training_batches = dataset.training_size // batch_size
        num_test_batches = dataset.test_size // batch_size
        
        best_val = np.inf
        best_training = np.inf
        best_epoch = 0
        
        rand_state = np.random.RandomState(seed=42)
        batch_idx = np.arange(num_training_batches)
        for epoch in range(epochs):
            batch_idx = rand_state.permutation(batch_idx)
            sys.stdout.flush()# Epoch Loop runs HM_EPOCHS TIMES
            # Epoch Loop runs HM_EPOCHS TIMES
            epoch_loss = 0  # initialize the epoch loss, this value will added up over all batches
            for idx in batch_idx:
                # LEARNING STEP
                epoch_x, epoch_y = dataset.get_batch(idx, data_set=DataSets.TRAINING)
                #pitches = pitch_model.predict(epoch_x)
                dirac = create_dirac(epoch_y[:,0])
                # current batch over the training
                reshaped_y = epoch_y[:,0]
                #print("Shape of Reshape",reshaped_y.shape)
                _, c, prediction = sess.run([self.train_op, self.loss,self.predict_op], feed_dict={self.x: epoch_x, self.y: reshaped_y,self.dirac: dirac})
                epoch_loss += c  # add batch loss too epoch loss
            
    
            print("Loss at Epoch", epoch + 1 ,"out of", epochs, "is", epoch_loss / num_training_batches)
            if (epoch_loss / num_training_batches) < best_training :
                best_training = epoch_loss / num_training_batches
            
            # DOCUMENTATION OF THE LEARNING SUCCESS
            
            if (epoch + 1) % 10 == 0:
                # WRITE VALIDATION SUCCESS EVERY 5´th epoch
                #pitches = pitch_model.predict(dataset.validation_x)
                #print("Pitches",pitches)
                dirac= create_dirac(dataset.validation_y[:,0])
                summary, c = sess.run([merged, self.loss], feed_dict={self.x: dataset.validation_x,self.dirac:dirac, self.y: dataset.validation_y[:,0]})
                test_writer.add_summary(summary, epoch)
                print('Valitdation Loss at step %s: %s' % (epoch + 1, c))
                if c < best_val :
                    best_val = c
                    #callback save better validation loss
                    print(self.save_dir)
                    self.saver.save(sess,self.save_dir,global_step=epoch)
                    best_epoch = epoch
                    print("Saves a session")
                
                # WRITE EVERY 10th EPOCH THE TRAINING SUCCESS
                training_x, training_y = dataset.get_batch(0, data_set=DataSets.TRAINING)
                dirac = create_dirac(training_y[:,0])
                summary, c, prediction = sess.run([merged, self.loss,self.predict_op], feed_dict={self.x: training_x,self.y: training_y[:,0],self.dirac: dirac})
                print("Predicts:", prediction, " for ", training_y[:,0])
                training_writer.add_summary(summary, epoch)
                
        print("======= TEST PERFORMANCE =======")
        
        # TEST PERFORMANCE
        predictions = []
        labels = []
        for idx in range(num_test_batches):
            test_x, test_y = dataset.get_batch(idx, data_set=DataSets.TEST)
            labels.extend(test_y)
            dirac = create_dirac(test_y[:,0])
            # adding other saver instruction for restoring the session and predict
            self.saver.restore(sess,self.save_dir+'-'+str(best_epoch))
            #self.saver.restore(sess,'./models/2048_2048_2048_64_exp2/model.ckpt-489')
            prediction = sess.run(self.predict_op,feed_dict={self.x:test_x,self.dirac:dirac})
            prediction = np.reshape(prediction,(batch_size,1))
            predictions.extend(prediction)
            for i in range(batch_size):
                print("Label:",test_y[i],"Prediction:",prediction[i])
        np.save('pitch_fc_deep_predictions.npy',predictions)
        np.save('pitch_fc_deep_labels.npy',labels)
        sess.close()
        return best_val, best_training

    def predict(self, x,model):
        """
        :param x: input x
        :return: Returns the prediction of the model for input x
        """
        
        with tf.Session() as session:
            # restore the model
            print(self.save_dir)
            #P = session.run(self.predict_op, feed_dict={self.x: x,self.dirac:diracs})
            #return P
