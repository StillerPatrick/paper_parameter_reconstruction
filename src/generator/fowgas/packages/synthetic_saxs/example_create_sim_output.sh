sigma=1.5  # sigma: softening length in, unit = 1 pixel
pitch=9.3  # pitch: period length of grating, unit = 1 pixel
fsize=4.2  # fsize: length of one grating feature, unit = 1 pixel
ff_angle=0.08  # ff_angle: rotation angle for detector image, unit = 1 radian
disp_y=19  # disp_y: displacement in y direction for detector image, unit = 1 pixel
disp_x=-198  # disp_x: displacement in x direction for detector image, unit = 1 pixel
nx=2048 # nx: x extent of detector image, unit = 1 pixel
ny=2048 # ny: y extent of detector image, unit = 1 pixel
psf_sigma=2.7  # psf_sigma: "blurring length" of detector, unit = 1 pixel
cutoff=1e10 # cutoff: maximum value detector pixel values will have, higher values will be cut off
write_fn=test.pickle # write filename

python simulate_saxs_data.py $sigma $pitch $fsize $ff_angle $disp_y $disp_x $nx $ny $psf_sigma $cutoff $write_fn
