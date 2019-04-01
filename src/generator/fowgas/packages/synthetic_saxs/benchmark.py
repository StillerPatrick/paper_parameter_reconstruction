import numpy as np
import time 

import simulate_saxs_data as simulator
from synthetic_saxs import SimulationPipeline as SimP
from synthetic_saxs import helpers
import h5py


def simulate_grating(pitch,fsize,sigma):
    p = simulator.SimulationParameters(sigma=sigma,fsize=fsize,pitch=pitch,number=1)
    sim_calc = simulator.SimulationCalculator(p)
    result = sim_calc.run()
    return result

#test for a single image 
begin = time.clock()
result = simulate_grating(pitch=512,fsize=120,sigma=1)
with h5py.File("endproduct.h5",'w') as hf:
    hf.create_dataset('ep',data= result.endproduct[1024,:],compression="gzip",compression_opts=9)

with h5py.File("dist.h5",'w') as hf:
    hf.create_dataset('dist',data= result.dist[1024,:],compression="gzip",compression_opts=9)

end = time.clock()
print("Time for a single:", str(end - begin)," Seconds")





#test for a single image 
begin = time.clock()
for i in range(100):
    result = simulate_grating(pitch=512,fsize=120+i,sigma=1)
    with h5py.File("endproduct.h5",'w') as hf:
        hf.create_dataset('ep',data= result.endproduct[1024,:],compression="gzip",compression_opts=9)

    with h5py.File("dist.h5",'w') as hf:
        hf.create_dataset('dist',data= result.dist[1024,:],compression="gzip",compression_opts=9)

end = time.clock()
print("Time for 100:", str(end - begin)," Seconds")



#test for a single image 
begin = time.clock()
for i in range(1000):
    result = simulate_grating(pitch=512,fsize=120+(i*0.1),sigma=1)
    with h5py.File("endproduct.h5",'w') as hf:
        hf.create_dataset('ep',data= result.endproduct[1024,:],compression="gzip",compression_opts=9)

    with h5py.File("dist.h5",'w') as hf:
        hf.create_dataset('dist',data= result.dist[1024,:],compression="gzip",compression_opts=9)

end = time.clock()
print("Time for 1000:", str(end - begin)," Seconds")
