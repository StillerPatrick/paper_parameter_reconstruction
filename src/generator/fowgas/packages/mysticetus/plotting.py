import numpy as np
from matplotlib import pyplot as plt
from . import model_driven_reco

def show_measurement_data(m, ncases, icase, fig, **kwargs):
    interpolation   = kwargs.pop('interpolation', 'nearest')
    origin          = kwargs.pop('origin', 'lower')
    cmap            = kwargs.pop('cmap', plt.cm.CMRmap)
    
    meas            = m.meas
    show_mask       = hasattr(m, 'mask')
    mask            = m.mask if show_mask else None
    show_dist       = hasattr(m, 'dist')
    dist            = m.dist if show_dist else None
    
#    aspect          = meas.shape[1]/meas.shape[0]
    aspect          = 1

    nsplts          = 1 + 2*int(show_mask) + int(show_dist)
    #nhsplts         = 1 if nsplts < 2 else 2
    #nvsplts         = (nsplts + nhsplts - 1) // 2
    nhsplts         = 1
    nvsplts         = nsplts
    
    isplt           = 0
    
    if show_dist:
        isplt   += 1
        fig.add_subplot(ncases*nvsplts, nhsplts, icase*nvsplts*nhsplts + isplt)
        im1             = np.log(dist)
        plt.imshow(im1, aspect=aspect, interpolation=interpolation, origin=origin, cmap=cmap)
        plt.title('True distribution')
    
    isplt   += 1
    fig.add_subplot(ncases*nvsplts, nhsplts, icase*nvsplts*nhsplts + isplt)
    im2             = np.log(meas)
    plt.imshow(im2, aspect=aspect, interpolation=interpolation, origin=origin, cmap=cmap)
    plt.title('Measurement')
    
    if show_mask:
        isplt   += 1
        fig.add_subplot(ncases*nvsplts, nhsplts, icase*nvsplts*nhsplts + isplt)
        im3         = mask
        plt.imshow(im3, aspect=aspect, interpolation=interpolation, origin=origin, cmap=cmap)
        plt.title('Mask')

        isplt   += 1
        fig.add_subplot(ncases*nvsplts, nhsplts, icase*nvsplts*nhsplts + isplt)
        im4         = mask*np.log(meas)
        plt.imshow(im4, aspect=aspect, interpolation=interpolation, origin=origin, cmap=cmap)
        plt.title('Mask * Measurement')

class ResultShow(object):
    def __init__(self, nresults, **kwargs):
        self.nresults = nresults
        self.fig      = self.figure(**kwargs)
        self.iresult  = 0
        self.nvsplts  = 9
        self.nhsplts  = 1
        self.isplt    = 0
        self.aspect   = kwargs['aspect']
        self.imshow_settings = {\
                               'interpolation': 'nearest',
                               'origin': 'lower',
                               'cmap': plt.cm.CMRmap}
        self.imshow_settings.update(**kwargs.pop('imshow_settings', {}))
        self.imshow_settings.update(**{'aspect':self.aspect})
    
    def __call__(self, result):
        self.isplt = 0
        self.show_all(result)
        self.iresult += 1
    
    def show_all(self, result):
        # plot of quality over basinhopping iterations
        sub = self.next_subplot()
        acc_minima = result['acc_minima']
        plt.plot(acc_minima)
        plt.title('Quality over BH iterations')
        
        # image of true distribution
        sub         = self.next_subplot()
        true_dist   = result['true_dist']
        if true_dist is not None:
            im1         = true_dist
            plt.imshow(im1, **self.imshow_settings)
            plt.title('True distribution')
        
        # image of optimized model distribution
        sub         = self.next_subplot()
        opt_dist    = result['opt_dist']
        im2         = opt_dist
        plt.imshow(im2, **self.imshow_settings)
        plt.title('Optimal Distribution')
        
        # plot of optimal distribution
        sub         = self.next_subplot()
        opt_dist    = result['opt_dist']
        p           = opt_dist[0,:]
        plt.plot(p)
        plt.xlim(0, opt_dist.shape[1])
        plt.ylim(opt_dist.min() - 0.1*(opt_dist.max()-opt_dist.min()),
                 opt_dist.max() + 0.1*(opt_dist.max()-opt_dist.min()))
        plt.title('Optimal Distribution')
        
        # log image of optimal dist's sim. measurement
        sub             = self.next_subplot()
        rel_norm        = 1
        if 'opt_params' in result.keys():
            if 'rn' in result['opt_params'].keys():
                rel_norm = result['opt_params']['rn']
        meas            = result['meas']
        mask            = result['mask']
        norm_meas       = np.sum(meas*mask) if mask is not None else np.sum(meas)
        normed_meas     = meas/norm_meas
        opt_meas        = result['opt_meas']
        norm_opt_meas   = np.sum(opt_meas*mask) if mask is not None else np.sum(opt_meas)
        normed_opt_meas = opt_meas/norm_opt_meas*rel_norm
        min_val         = normed_meas.min()
        print('min_val: ', min_val)
        max_val         = normed_meas.max()
        print('max_val: ', max_val)
        im3             = np.log(normed_opt_meas)
        plt.imshow(im3, **{**self.imshow_settings, 'vmin':-16, 'vmax':np.log(max_val)})
        #im3             = np.log(opt_meas)
        #plt.imshow(im3, **self.imshow_settings)
        plt.title('Optimal Measurement')
        
        # log image of true measurement
        sub             = self.next_subplot()
        im4             = np.log(normed_meas)
        plt.imshow(im4, **{**self.imshow_settings, 'vmin':-16, 'vmax':np.log(max_val)})
        #im4             = np.log(normed_meas)
        #plt.imshow(im4, **{**self.imshow_settings, 'vmin':np.log(min_val), 'vmax':np.log(max_val)})
        #im4             = np.log(meas)
        #plt.imshow(im4, **self.imshow_settings)
        plt.title('Measurement')
        
        # plot of true and optimal dist's sim. measurement
        sub = self.next_subplot()
        if mask is not None:
            sub.plot(       (normed_meas*    mask)[normed_meas.    shape[0]//2,:], label='Measurement')
            sub.plot(       (normed_opt_meas*mask)[normed_opt_meas.shape[0]//2,:], label='Optimal Measurement')
            ymax = np.max([((normed_meas    *mask)[normed_meas.    shape[0]//2,:]).max(),
                           ((normed_opt_meas*mask)[normed_opt_meas.shape[0]//2,:]).max()])
            sub.fill_between(
                range(normed_opt_meas.shape[1]),
                ymax * (1-mask)[normed_opt_meas.shape[0]//2,:],
                color='red',
                alpha=0.3
            )
            sub.set_xlim(0, normed_meas.shape[1])
            sub.legend()
            sub.set_title('Measurement and Optimal Measurement, center line')
        else:
            sub.plot(       (normed_meas)    [normed_meas.    shape[0]//2,:], label='Measurement')
            sub.plot(       (normed_opt_meas)[normed_opt_meas.shape[0]//2,:], label='Optimal Measurement')
            ymax = np.max([((normed_meas    )[normed_meas.    shape[0]//2,:]).max(),
                           ((normed_opt_meas)[normed_opt_meas.shape[0]//2,:]).max()])
            sub.set_xlim(0, meas.shape[1])
            sub.legend()
            sub.set_title('Measurement and Optimal Measurement, center line')
        
        # log plot of true and optimal dist's sim. measurement
        sub = self.next_subplot()
        if mask is not None:
            sub.plot(       (normed_meas*    mask)[normed_meas.    shape[0]//2,:], label='Measurement')
            sub.plot(       (normed_opt_meas*mask)[normed_opt_meas.shape[0]//2,:], label='Optimal Measurement')
            ymax = np.max([((normed_meas    *mask)[normed_meas.    shape[0]//2,:]).max(),
                           ((normed_opt_meas*mask)[normed_opt_meas.shape[0]//2,:]).max()])
            sub.fill_between(
                range(normed_opt_meas.shape[1]),
                ymax * (1-mask)[normed_opt_meas.shape[0]//2,:],
                color='red',
                alpha=0.3
            )
            sub.set_xlim(0, normed_meas.shape[1])
            sub.set_yscale('log')
            sub.legend()
            sub.set_title('Measurement and Optimal Measurement, center line')
        else:
            sub.plot(       (normed_meas)    [normed_meas.    shape[0]//2,:], label='Measurement')
            sub.plot(       (normed_opt_meas)[normed_opt_meas.shape[0]//2,:], label='Optimal Measurement')
            ymax = np.max([((normed_meas    )[normed_meas.    shape[0]//2,:]).max(),
                           ((normed_opt_meas)[normed_opt_meas.shape[0]//2,:]).max()])
            sub.set_xlim(0, meas.shape[1])
            sub.set_yscale('log')
            sub.legend()
            sub.set_title('Measurement and Optimal Measurement, center line')
        
        # image of mask
        sub = self.next_subplot()
        if mask is not None:
            im5         = mask
            plt.imshow(im5, **self.imshow_settings)
            plt.title('Mask')
    
    def next_subplot(self):
        self.isplt   += 1
        sub = self.fig.add_subplot(self.nresults*self.nvsplts,
                                   self.nhsplts,
                                   self.iresult*self.nvsplts*self.nhsplts + self.isplt)
        return sub
    
    def figure(self, **kwargs):
        figsize = kwargs['figsize']
        return plt.figure(figsize=figsize)
