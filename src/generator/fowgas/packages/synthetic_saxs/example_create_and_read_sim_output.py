from synthetic_saxs.simulate_saxs_data import main
import pickle
import time 
# Set up "command line arguments"
argv = [\
    'main', # program name - in true command line operation, this argument is passed implicitly
    '4.9406564584124654e-324',  # sigma: softening length in, unit = 1 pixel
    '176.0',  # pitch: period length of grating, unit = 1 pixel
    '56.6666',  # fsize: length of one grating feature, unit = 1 pixel
    '0.00', # ff_angle: rotation angle for detector image, unit = 1 radian
    '0',   # disp_y: displacement in y direction for detector image, unit = 1 pixel
    '0', # disp_x: displacement in x direction for detector image, unit = 1 pixel
    '2048', # nx: x extent of detector image, unit = 1 pixel
    '2048', # ny: y extent of detector image, unit = 1 pixel
    '2.7',  # psf_sigma: "blurring length" of detector, unit = 1 pixel
    '1e10', # cutoff: maximum value detector pixel values will have, higher values will be cut off
    'test.pickle' # write filename
]

# Execute the simulation script
start = time.time()
main(argv)
end = time.time()
print(end - start)
# Retrieve the simulation result object from file
fn = argv[11]
with open(fn, 'rb') as file:
    test_result = pickle.load(file)
