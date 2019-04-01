import numpy as np
import sys
import getopt
import os
import pickle
import copy
from matplotlib import pyplot as plt

from synthetic_saxs import SimulationPipeline as SimP
from synthetic_saxs import helpers


def generateSimulationParameters(N):
    result = []
    nx = 2048
    numParameters = round(N ** (1/3));
    number = 0
    for pitch in np.linspace(64,512,46):
        for fsize in np.linspace(pitch*0.25,pitch*0.75,46):
            for sigma in np.linspace(1e-9,(pitch-fsize)/4,46):
                result.append(SimulationParameters(sigma=sigma,pitch=pitch,fsize=fsize,number=number))
                number += 1
    return np.array(result)




class SimulationParameters():
    def __init__(self,
                 sigma,
                 pitch,
                 fsize,
                 number,
                 oversampling_factor=1.0,
                 nx = 2048,
                 ny = 2048,
                 psf= 0.0,
                 cutoff=np.infty,
                 ff_angle=0,
                 disp_y=0,
                 disp_x=0):
        # Due to later rotation and displacement of the detector image, more frequencies need to be simulated then the number that will be in the output images. Therefor the resolution of the real space also needs to be higher. The oversampling_factor is: [pixel number along one axis that needs to be sim.d]/[pixel number along one axis that will be in the output]. At the moment it's assumed to be the same for both axes and it provides enough additional frequencies to allow for
        # * one arbitrary rotation, followed by
        # * a displacement along both axis of at most the half output pixel number.
        self.__dict__['oversampling_factor'] = oversampling_factor
        self.__dict__['nx'] = nx
        self.__dict__['ny'] = ny
        self.__dict__['psf_sigma'] = psf
        self.__dict__['cutoff'] = cutoff
        self.__dict__['sigma'] = sigma
        self.__dict__['pitch'] = pitch
        self.__dict__['fsize'] = fsize
        self.__dict__['ff_angle'] = ff_angle
        self.__dict__['disp_y'] = disp_x
        self.__dict__['disp_x'] = disp_y
        self.__dict__['number'] = number


class SimulationCalculator(object):
    def __init__(self, simParams):
        self.simParams = simParams.__dict__
    
    def run(self):
        params = { \
            'sigma': self.simParams['sigma'] * self.simParams['oversampling_factor'],
            'pitch': self.simParams['pitch'] * self.simParams['oversampling_factor'],
            'fsize': self.simParams['fsize'] * self.simParams['oversampling_factor'],
            'ff_angle': self.simParams['ff_angle'],
            'disp': self.create_disp()
        }
        
        consts = { \
            'domain': self.create_domain(),
            'crop_slice': self.create_crop_slice(),
            'photon_statistics': self.create_photon_statistics(),
            'psf': self.create_psf(),
            'cutoff': self.simParams['cutoff']
        }
        
        SimP.pl2(params, SimP.interims, consts)
        return copy.deepcopy(SimP.savings)

    def create_domain(self):
        xres = self.simParams['nx'] * self.simParams['oversampling_factor']
        yres = self.simParams['ny'] * self.simParams['oversampling_factor']
        x = np.linspace(0, xres, xres, endpoint=False)
        y = np.linspace(0, xres, yres, endpoint=False)
        xx, yy = np.meshgrid(x, y)
        return {'x': x, 'y': y, 'xx': xx, 'yy': yy}
    
    def create_crop_slice(self):
        crop_start = 0.5 * (self.simParams['oversampling_factor'] - 1)
        crop_end = 0.5 * (self.simParams['oversampling_factor'] + 1)
        return (slice(int(crop_start * self.simParams['ny']), int(crop_end * self.simParams['ny'])),slice(int(crop_start * self.simParams['nx']), int(crop_end * self.simParams['nx'])))
    
    def create_psf(self):
        domain = self.create_domain()
        return helpers.create_psf(domain['xx'], domain['yy'], self.simParams['psf_sigma'])
    
    def create_disp(self):
        return (self.simParams['disp_y'], self.simParams['disp_x'])
    
    def create_photon_statistics(self):
        return lambda x: x


class Usage(Exception):
    def __init__(self, msg):
        self.msg = msg


def main(argv=None):
    if argv is None:
        argv = sys.argv
    try:
        try:
            opts, args = getopt.getopt(argv[1:], 'h', ['help',])
        except getopt.error as msg:
            raise Usage(msg)
    except Usage as err:
        print(err.msg, file=sys.stderr)
        print('For help use --help', file=sys.stderr)
        return 2
    
    # setup simulation parameters
    sim_params = SimulationParameters(\
                                      sigma =     float(args[ 0]),
                                      pitch =     float(args[ 1]),
                                      fsize =     float(args[ 2]),
                                      ff_angle =  float(args[ 3]),
                                      number = 1,
                                      disp_y =    int(  args[ 4]),
                                      disp_x =    int(  args[ 5]),
                                      nx =        int(  args[ 6]),
                                      ny =        int(  args[ 7]),
                                      psf =       float(args[ 8]),
                                      cutoff =    float(args[ 9])
                                      )
    writefn = str(args[10])
    
    # setup and run simulation
    sim_calc = SimulationCalculator(sim_params)
    result = sim_calc.run()
    
    # write output
    if os.path.exists(writefn):
        raise FileExistsError('Error: path exists!')
    with open(writefn, 'wb') as file:
        pickle.dump(result, file, protocol=pickle.HIGHEST_PROTOCOL)


if __name__=="__main__":
    sys.exit(main())
