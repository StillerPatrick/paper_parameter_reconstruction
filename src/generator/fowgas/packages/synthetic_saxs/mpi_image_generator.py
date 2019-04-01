from  mpi4py import MPI
import numpy  as np
import sys
import os
import pickle
import copy
import simulate_saxs_data as simulator 
from synthetic_saxs import SimulationPipeline as SimP
from synthetic_saxs import helpers
import h5py
import time 
#get mpi enviorment
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


#load commandline parameters
argv = sys.argv
n = int(argv[1])
prefix = argv[2]

#check the enviorment and set local enviorment
local_n = 0
if rank == 0:
    if n % size != 0 :
        print( "Number of processes must devide number of images without rest")
    else:
        local_n = [ n // size ]
        print('enviorment check completed with scatter-factor: '+ str(local_n))

local_n = comm.bcast(local_n, root=0)

#process with rank 0 create dataset-paramters
params = np.arange(n,dtype=np.float64)
np_chunks = np.arange(size,dtype=np.float64)
if rank == 0:
    params = simulator.generateSimulationParameters(n) 
    #save the labels as a seperate file  
    labels = []
    for i,label in enumerate(params):
        labels.append(label.__dict__)
    np_labels = np.array(labels)
    labelpath = os.path.join(prefix,'labels')
    np.save(labelpath,np_labels)
    #create chunks, cause comm.Scatter does not work with nested datatypes
    chunks = [[] for _ in range(size)]
    for i,chunk in enumerate(params):
        chunks[i % size].append(chunk)
    np_chunks = np.array(chunks)
else:
    params = None
    np_chunks = None

#scatter the dataset with mpi.scatter
comm.barrier()
start = MPI.Wtime()
local_data= comm.scatter(np_chunks,root = 0)
# start parallel zone
for p in local_data:
    print(len(local_data))
    #create filename
    writefn_tailored = os.path.join(prefix,'{:06d}'.format(p.__dict__['number']) + '_dist.h5')
    writefn_endproduct = os.path.join(prefix,'{:06d}'.format(p.__dict__['number']) + '_endproduct.h5')
    
    sim_calc = simulator.SimulationCalculator(p)
    result = sim_calc.run()

    with h5py.File(writefn_tailored,'w') as hf :
        ds = hf.create_dataset('dist',data=result.dist[1024,:], compression="gzip", compression_opts=9)

    with h5py.File(writefn_endproduct,'w') as hf :
        ds = hf.create_dataset('endproduct',data=result.endproduct[1024,:],compression="gzip", compression_opts=9)
#end parallel zone
comm.barrier()
end = MPI.Wtime()
MPI.Finalize()
#if rank == 0:
    #print("Runtime of parallel zone", str(end-begin))
