import time
import datetime
import csv
import os.path
import numpy as np
from DataSetSigmaFsizePitch import ScatterImageSet
from Enumerations import DataSets

from argparse import ArgumentParser

from Dataset import DataSet
from CNN import PITCH_CNN


def create_hyperparameter_string(num_layers,
                                 filter_size,
                                 num_filters,
                                 residual_distance,
                                 num_epochs,
                                 batch_size,
                                 num_dense,
                                 learning_rate,
                                 normalize,
                                 shift):
    arguments = locals()
    result_string = ""
    # create string from locals
    for keys, value in arguments.items():
	
       	"create string key2=value2,key2=value2,..."
        result_string = result_string + keys + "=" + str(value) + ","

        # add timestamp
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y_%m_%d_%H_%M')
    result_string = result_string + "time=" + st
    return result_string


def write_in_csv(num_layers,
                 num_images,
                 filter_size,
                 num_filters,
                 residual_distance,
                 num_epochs,
                 batch_size,
                 num_dense,
                 learning_rate,
                 val_loss,
                 training_loss,
                 time,
                 model_path,
                 normalize,
                 shift):
    hyperparams = locals()
    filename = "session.csv"
    fileEmpty = os.stat(filename).st_size == 0
    with open(filename, 'a') as csvfile:
        headers = hyperparams.keys()
        writer = csv.DictWriter(csvfile, headers, delimiter=',')
        if fileEmpty:
            writer.writeheader()
        writer.writerow(hyperparams)


def main():


    parser = ArgumentParser()

    parser.add_argument("--layers", dest="num_layers", type=int)
    parser.add_argument("--filter", dest="filter_size", type=int)
    parser.add_argument("--nfilter", dest="num_filters", type=int)
    parser.add_argument("--rdes", dest="residual_distance", type=int)
    parser.add_argument("--epochs", dest="num_epochs", type=int)
    parser.add_argument("--batchsize", dest="batch_size", type=int)
    parser.add_argument("--dense", dest="num_dense_units", type=int)
    parser.add_argument("--lr", dest="learning_rate", type=float)
    parser.add_argument("--normalize", dest="normalize", type=int)
    parser.add_argument("--shift", dest="shift", type=int)
    parser.add_argument("--restore",dest="restore",default="",type=str)
    
    
    


    args = parser.parse_args()
    num_training_images = 0
    dataset = ScatterImageSet(x_path='/bigdata/hplsim/production/aipr/pitch_set/images',
                              y_path='/bigdata/hplsim/production/aipr/pitch_set/labels.npy',
                              batch_size= args.batch_size,
                              training_size=num_training_images,
                              test_size= 70000,
                              validation_size=0,
                              normalize = bool(args.normalize),
                              shift = bool(args.shift)
                              )

    # 5. Create Hyperparamter String
    # 6. Create CNN
    
    print(args)
   
   
    hpstring = create_hyperparameter_string(args.num_layers,
                                            args.filter_size,
                                            args.num_filters,
                                            args.residual_distance,
                                            args.num_epochs,
                                            args.batch_size,
                                            args.num_dense_units,
                                            args.learning_rate,
                                            args.normalize,
                                            args.shift)
                                            
    print(hpstring) 
    tensorboard =  "training_sessions/"+ hpstring
    savedir = "models/"+ hpstring + '/model.ckpt'
    
    
    if args.restore != "":
        model = PITCH_CNN(args.restore,
                          tensorboard,
                          args.num_layers,
                          args.filter_size,
                          args.num_filters,
                          args.residual_distance,
                          args.num_dense_units,
                          args.learning_rate)
    else:
        model = PITCH_CNN(savedir,
                          tensorboard,
                          args.num_layers,
                          args.filter_size,
                          args.num_filters,
                          args.residual_distance,
                          args.num_dense_units,
                          args.learning_rate)
    
                    
    # 7. Fit Neuronal Network
    start_time = time.time()
    best_val,best_training = model.fit(args.num_epochs, args.batch_size, dataset,"")
    end_time = time.time()
    time_elapsed = (end_time - start_time)/60.
    write_in_csv(args.num_layers,
                 num_training_images,
                 args.filter_size,
                 args.num_filters,
                 args.residual_distance,
                 args.num_epochs,
                 args.batch_size,
                 args.num_dense_units,
                 args.learning_rate,
                 best_val,
                 best_training,
                 time_elapsed,
                 savedir,
                 args.normalize,
                 args.shift)


if __name__ == "__main__":
    main()





# 8. Test Neuronal Network, Write Error to CSV File
