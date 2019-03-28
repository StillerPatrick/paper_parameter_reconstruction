
num_pitches, num_fsizes, num_sigmas = 64
for pitch in np.linspace(64,512,num=num_pitches):
    for featureSize in np.linspace(0.25*pitch,0.75*pitch,num=num_fsizes):
        for sigma in np.linspace(1e-9,(pitch-featureSize)/4.,num=num_sigmas):
            add_to_P(sigma,pitch,fsize,number))
            
