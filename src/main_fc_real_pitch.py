import time
import datetime
import csv
import os.path
import numpy as np
from DataSetSigmaFsizePitch import ScatterImageSet
from Enumerations import DataSets

from argparse import ArgumentParser

from Dataset import DataSet
from fc_real_pitch import FC



def main():


    parser = ArgumentParser()
    parser.add_argument("--numTrain", dest="num_images_train", type=int)
    parser.add_argument("--numValid", dest="num_images_validation", type=int)
    parser.add_argument("--numTest", dest="num_images_test", type=int)
    parser.add_argument("--numEpochs",dest="num_epochs",type=int)
    parser.add_argument("--batchSize",dest="batch_size",type=int)
    parser.add_argument("--model",dest="model",default="",type=str)
    parser.add_argument("--lr",dest="learning_rate",type=float)
    parser.add_argument("--name",dest="name",type=str)
   
    
    args = parser.parse_args()
  
    dataset = ScatterImageSet(x_path='/bigdata/hplsim/production/aipr/extended_pitch_set/images',
                              y_path='/bigdata/hplsim/production/aipr/extended_pitch_set/labels.npy',
                              batch_size= args.batch_size,
                              training_size=args.num_images_train,
                              test_size= args.num_images_test,
                              validation_size=args.num_images_validation,
                              normalize = False,
                              shift = False
                              )


    # 5. Create Hyperparamter String
    # 6. Create CNN
    

    tensorboard =  "training_sessions/"+ args.name
    savedir = './models/'+args.name +'/model.ckpt'
    
    
    fc_model= FC(savedir,tensorboard,args.learning_rate)
              
    # 7. Fit Neuronal Network
    start_time = time.time()
    best_val,best_training = fc_model.fit(args.num_epochs, args.batch_size, dataset,args.model)
    end_time = time.time()
    time_elapsed = (end_time - start_time)/60.
    print("Best Validation",best_val)
    print("Best Training",best_training)
    print("In time:",time_elapsed)



if __name__ == "__main__":
    main()





# 8. Test Neuronal Network, Write Error to CSV File
