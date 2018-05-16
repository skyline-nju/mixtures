"""
Study phase seperation of liquid mixtures by solving PDEs.

@file reaction_diffusion.py
@author skyline-nju
@date 2018-05-13
"""

import matplotlib.pyplot as plt
import numpy as np


def load_h5(filename):
    import h5py
    f = h5py.File(filename, 'r')
    t = f['1']['t']
    x = f['1']['x']
    y = f['1']['y']
    phi = f['1']['Phi']
    # v = f['1']['V']
    for i in range(t.size):
        print(t[i], np.mean(phi[i]), np.max(phi[i]), np.min(phi[i]))
        plt.clf()
        plt.pcolormesh(x, y, phi[i])
        plt.colorbar()
        plt.axis("image")
        plt.title(r"$t=%.3f$" % t[i])
        plt.show()
        plt.pause(0.1)


if __name__ == "__main__":
    plt.ion()
    load_h5("droplet.r0_2.h5")
