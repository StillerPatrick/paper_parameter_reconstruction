# import packages
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import ticker as mtick
from scipy import special

# append path for loading own packages
import sys
sys.path.append("../packages")

# import own packages
import edges
import grating
import copy
from auxiliary import theta
from auxiliary import evp
from auxiliary import transpose_extent
import auxiliary as aux
import jitter
import density
import imaging as imgg
from imaging import expand_coordinates

class Edge:
    def __init__(self, function, name):
        self.function = function
        self.name = name

def validation_plot(yp, edge_fct, feature_fct, grating_fct, zp, jitter_fct):
#def validation_plot(feature, grating_function, jitter_function, parameter_string, **params):
    fig = plt.figure(figsize=(20,20))
    
    sub1 = fig.add_subplot(3, 3, 1, adjustable='datalim', aspect=1.)
    plt.plot(yp, edge_fct(yp))
    #plt.plot(evp(params['yp']), evp(params['edge']).function(evp(params['yp'])))
    plt.title('Edge function')
    plt.xlabel('Target y coordinate [m]')
    plt.ylabel('Target x coordinate [m]')
    sub1.get_xaxis().set_major_formatter(mtick.FormatStrFormatter('%.1e'))
    sub1.get_yaxis().set_major_formatter(mtick.FormatStrFormatter('%.1e'))
    
    sub2 = fig.add_subplot(3, 2, 2, adjustable='datalim', aspect=1.)
    plt.plot(yp, feature_fct(yp))
    #plt.plot(evp(params['yp']), feature(evp(params['yp'])))
    plt.title('Feature function')
    plt.xlabel('Target y coordinate [m]')
    plt.ylabel('Target x coordinate [m]')
    sub2.get_xaxis().set_major_formatter(mtick.FormatStrFormatter('%.1e'))
    sub2.get_yaxis().set_major_formatter(mtick.FormatStrFormatter('%.1e'))
    
    sub3 = fig.add_subplot(3, 2, 3, adjustable='datalim', aspect=1.)
    plt.plot(yp, grating_fct(yp))
    plt.title('Grating function')
    plt.xlabel('Target y coordinate [m]')
    plt.ylabel('Target x coordinate [m]')
    sub3.get_xaxis().set_major_formatter(mtick.FormatStrFormatter('%.1e'))
    sub3.get_yaxis().set_major_formatter(mtick.FormatStrFormatter('%.1e'))
    
    sub4 = fig.add_subplot(3, 2, 4, adjustable='datalim', aspect=1.)
    plt.plot(jitter_fct(zp), zp)
    plt.title('Jitter function')
    plt.xlabel('Target y coordinate [m]')
    plt.ylabel('Target z coordinate [m]')
    sub4.get_xaxis().set_major_formatter(mtick.FormatStrFormatter('%.1e'))
    sub4.get_yaxis().set_major_formatter(mtick.FormatStrFormatter('%.1e'))
    
    return fig

# Plot summed density
def summed_density_plot(dd_sum, extent):
    fig = plt.figure(figsize=(20, 6))
    sub1 = fig.add_subplot(1, 1, 1)
    
    aux.imshow(dd_sum, extent=extent)
    sub1.get_xaxis().set_major_formatter(mtick.FormatStrFormatter('%.1e'))
    sub1.get_yaxis().set_major_formatter(mtick.FormatStrFormatter('%.1e'))
    plt.title('xy plot summed slices')
    plt.colorbar()
    return fig

def density_left_plot(dd_zx, extent):
    fig = plt.figure(figsize=(20, 6))
    sub1 = fig.add_subplot(1, 1, 1)
    
    aux.imshow(dd_zx[:,::-1].T, extent=extent)
    return fig

# Plot slices of 3D density
def density_slices_plot(ddd, extent):
    fig = plt.figure(figsize=(20, 6))
    
    sub1 = fig.add_subplot(1, 3, 1)
    aux.imshow(ddd[:, :, ddd.shape[2]/8].T, extent = extent, origin='upper')
    sub1.get_xaxis().set_major_formatter(mtick.FormatStrFormatter('%.1e'))
    sub1.get_yaxis().set_major_formatter(mtick.FormatStrFormatter('%.1e'))
    plt.title('xy plot at 1/8 of z range')
    plt.colorbar()

    sub2 = fig.add_subplot(1, 3, 2)
    aux.imshow(ddd[:, :, ddd.shape[2]/2].T, extent = extent, origin='upper')
    sub2.get_xaxis().set_major_formatter(mtick.FormatStrFormatter('%.1e'))
    sub2.get_yaxis().set_major_formatter(mtick.FormatStrFormatter('%.1e'))
    plt.title('xy plot at 1/2 of z range')
    plt.colorbar()
    
    sub3 = fig.add_subplot(1, 3, 3)
    aux.imshow(ddd[:, :, ddd.shape[2]/8*7].T, extent = extent, origin='upper')
    sub3.get_xaxis().set_major_formatter(mtick.FormatStrFormatter('%.1e'))
    sub3.get_yaxis().set_major_formatter(mtick.FormatStrFormatter('%.1e'))
    plt.title('xy plot at 7/8 of z range')
    plt.colorbar()
    
    return fig

def dens2D (x, y, z, density_function, kernelx, kernely, T):
    kernelx_size = kernelx.shape[0]
    kernely_size = kernely.shape[1]
    
    xnew = expand_coordinates(x, kernelx_size)
    ynew = expand_coordinates(y, kernely_size)
    
    xx, yy = np.meshgrid(xnew, ynew)
    xx, yy = xx.T, yy.T
    
    print('x.size: {}'.format(x.size))
    print('y.size: {}'.format(y.size))
    print('kernelx_size: {}'.format(kernelx_size))
    print('kernely_size: {}'.format(kernely_size))
    print('xnew.size: {}'.format(xnew.size))
    print('ynew.size: {}'.format(ynew.size))
    
    dd_zx  = np.zeros(shape=(0, x.size))
    dd_sum = np.zeros(shape=(x.size, y.size))
    for this_z in z:
        xxp, yyp, zp = aux.rcoords2tcoords(xx, yy, this_z)
        xxpp, yypp, zpp = T(xxp, yyp, zp)
        #xxp, yyp, zp = T(xx, yy, this_z)
        #xxpp, yypp, zpp = aux.rcoords2tcoords(xxp, yyp, zp)
        
        dd = density_function(xxpp, yypp, zpp)
        dd, dde = imgg.smooth(dd, kernelx, [xnew.min(), xnew.max(), ynew.min(), ynew.max()])
        dd, dde = imgg.smooth(dd, kernely, dde)
        dd_sum += dd
        dd_zx = np.concatenate((dd_zx, dd[:,0:1].T), axis=0)
    return dd_sum, dd_zx

def literal_param_string(param, name):
    """
    Return string that represents literal parameter information.
    
    Arguments:
    -----
    param = literal value of parameter
    name  = string, indicates type of parameter
    
    Return:
    -----
    Parameter string.
    """
    string = param + '-' + name
    return string

def numerical_param_string(param, factor, unit, name):
    """
    Return string that represents numerical parameter information.
    
    Arguments:
    -----
    param  = numerical value of parameter
    factor = factor to multiply 'param' with to get value in unit indicated by 'unit'
    unit   = string, indicates unit of represented parameter
    name   = string, indicates type of parameter
    
    Return:
    -----
    Parameter string.
    """
    string = '{:03.0f}'.format(param * factor) + unit + '-' + name
    return string

def gen_paramstring(**params):
    paramstring = ''
    paramstring += numerical_param_string(params['edge_width'], 1e9, 'nm', 'edgewidth')
    paramstring += ('_' + literal_param_string(params['edge'].name, 'edgetype'))
    paramstring += ('_' + numerical_param_string(params['pitch'], 1e9, 'nm', 'pitch'))
    paramstring += ('_' + numerical_param_string(params['jitter_amplitude'], 1e9, 'nm', 'jampl'))
    paramstring += ('_' + numerical_param_string(params['jitter_wavelength'], 1e9, 'nm', 'jwl'))
    paramstring += ('_' + numerical_param_string(params['sigmax'], 1e9, 'nm', 'sigma'))
    paramstring += ('_' + numerical_param_string(params['tilt'], 180/np.pi, 'deg', 'tilt'))
    paramstring += ('_' + numerical_param_string(params['edge_angle'], 180/np.pi, 'deg', 'edgeangle'))
    return paramstring

def run(**params):
    """
    Calculate a 2D density distribution and possibly write it and other output for validation to file.
    
    Arguments:
    
    
    """
    parameter_string = gen_paramstring(**params)
    print(parameter_string)
    
    # Make density function
    feature          = grating.symm_feat(       evp(params['hfs']),
                                                evp(params['edge']).function)
    
    grating_function = grating.feature_grating( evp(params['pitch']),
                                                feature)
    
    jitter_function  = jitter.make_jitter(      evp(params['jitter_amplitude']),
                                                evp(params['jitter_wavelength']))
    
    density_function = density.make_density(    grating_function,
                                                jitter_function)
    
    T = aux.make_tcoords2ttcoords(evp(params['tilt']))
    
    # Plot functions for validation
    if params['validation_plot'] == True:
        fig_vp = validation_plot(params['yp'], params['edge'].function, feature, grating_function, params['zp'], jitter_function)
        path = evp(params['outdir']) + '/' + parameter_string + "_validate.pdf"
        if not aux.path_exists(path, mode='Fail'):
            fig_vp.savefig(path)
        plt.close(fig_vp)
    
    # Calculate and sum density slices along z axis
    kernelx = evp(params['kernelx'])
    kernelx = kernelx/kernelx.sum()
    
    kernely = evp(params['kernely'])
    kernely = kernely/kernely.sum()
    
    x       = evp(params['x'])
    y       = evp(params['y'])
    z       = evp(params['z'])
    dd_sum, dd_zx  = dens2D(x, y, z, density_function, kernelx, kernely, T)
    dd_sum  = dd_sum * evp(params['nelectrons'])
    dd_zx   = dd_zx * evp(params['nelectrons'])
    
    # Save summed density to file
    if evp(params['save_summed_density']) == True:
        path = evp(params['outdir']) + "/" + parameter_string + "_summeddensity.txt"
        if not aux.path_exists(path, mode='Fail'):
            np.savetxt(path, dd_sum)
    
    # Plot summed density
    if params['summed_density_plot'] == True:
        fig_sd = summed_density_plot(dd_sum, transpose_extent(params['extent_laser_xy']))
        path = evp(params['outdir']) + "/" + parameter_string + "_summeddensity_plot.pdf"
        if not aux.path_exists(path, mode='Fail'):
            fig_sd.savefig(path)
        plt.close(fig_sd)
    
    # Plot left view on density
    fig_lv = density_left_plot(dd_zx, [evp(params['zmin']), evp(params['zmax']), evp(params['xmax']), evp(params['xmin'])])
    
    # 3D density grid
    if evp(params['calc3d']):
        # Calculate 3D density grid
        xxx, yyy, zzz = np.meshgrid(evp(params['x']), evp(params['y']), evp(params['z']))
        xxxp, yyyp, zzzp = aux.rcoords2tcoords(xxx, yyy, zzz)
        ddd = evp(params['nelectrons']) * density_function(xxxp, yyyp, zzzp)
        
        # Plot slices of 3D density
        extent = transpose_extent(evp(params['extent_laser_xy']))
        fig_ds = density_slices_plot(ddd, extent)