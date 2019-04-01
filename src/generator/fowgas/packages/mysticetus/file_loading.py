import numpy as np
from PIL import Image
import csv

from . import model_driven_reco



def load_measurement(file_name):
    """
    Load measurement image data and return it as numpy array. For example for tif images.
    """
    return np.array(Image.open(file_name))



def load_mask(file_name):
    """
    Load mask image data and return it as numpy array. For example for tif images.
    """
    return np.array(model_driven_reco.bisect(np.array(Image.open(file_name))), dtype=float)



def mask_exists(shot_name):
    """
    Return True if a mask image can be opened for the given shot name, else return False.
    """
    try:
        Image.open(shot_name+'_mask.tif')
    except FileNotFoundError:
        return False
    return True



def load_endpoint_info(file_name):
    """
    Load endpoint table file and return it as dict of lists.
    """
    endpoint_info = {'name':[], 'centerX':[], 'centerY':[], 'endX':[], 'endY':[], 'otherX':[], 'otherY':[]}
    with open(file_name, 'r') as csvfile:
        reader = csv.DictReader(csvfile, fieldnames=['name', 'centerX', 'centerY', 'endX', 'endY', 'otherX', 'otherY'], delimiter='\t')
        for row in reader:
            name        = row['name']
            centerX_f   = float(row['centerX'])
            centerY_f   = float(row['centerY'])
            endX_f      = float(row['endX'])
            endY_f      = float(row['endY'])
            otherX_f    = float(row['otherX'])
            otherY_f    = float(row['otherY'])
            
            endpoint_info['name'].   append(name)
            endpoint_info['centerX'].append(centerX_f)
            endpoint_info['centerY'].append(centerY_f)
            endpoint_info['endX'].   append(endX_f)
            endpoint_info['endY'].   append(endY_f)
            endpoint_info['otherX']. append(otherX_f)
            endpoint_info['otherY']. append(otherY_f)
            
    return endpoint_info
