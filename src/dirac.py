import numpy as np

def single_dirac(pitch):
    return np.array([ int(i % pitch == 0) for i in range(2048)])

def create_dirac(pitch_array):
    result =[]
    for pitch in pitch_array:
        result.append(single_dirac(pitch))
    return np.array(result)
