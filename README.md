# Parameter Reconstruction with Deep Learning

## Requirements
- Python (>= 3.5)
- NumPy (>= 1.11.0)
- Scipy (>= 0.17.0)
- Tensorflow-gpu (>= 1.11.0)
- Matplotlib (>= 3.0.2)
- openMPI (>= 2.0.2)
- mpi4py (>= 2.0.1)
- h5py (>= 2.7.1)
- gcc (>= 5.3.0)

## Before you start
```
$ git clone https://github.com/StillerPatrick/paper_parameter_reconstuction.git
$ cd paper_parameter_reconstruction
```

## Generate the Dataset
The generator is embedded in the [fowgas enviorment](https://github.com/ComputationalRadiationPhysics/fowgas) from malte zacharias. For generating the database use following commands: 

Before you start the generation of the database you have to edit the start_generation.sh shell script. The script offers you three variables to configure your generator:

```
TARGET = '/path/for/the/images'
NUMIMAGES = 100000 #Number of images
PROCESSES = 64 #Number of parallel Processes 
```

Its important that you create the folder for the images before. 

```
$ cd src
$ mkdir path/to/the/images
$ source start_generator.sh
```


## Train the model 



## Inference 



