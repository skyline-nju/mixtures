"""
Study phase seperation of liquid mixtures by solving PDEs.

@file reaction_diffusion.py
@author skyline-nju
@date 2018-05-13
"""

import matplotlib.pyplot as plt
import numpy as np
import os


def load_h5(filename):
    import h5py
    f = h5py.File(filename, 'r')
    t = f['1']['t']
    # x = f['1']['x']
    # y = f['1']['y']
    phi = f['1']['Phi']
    # v = f['1']['V']
    for i in range(t.size):
        print(t[i], np.mean(phi[i]), np.max(phi[i]), np.min(phi[i]))
        plt.clf()
        plt.subplot(121)
        # plt.pcolormesh(x, y, phi[i])
        plt.imshow(phi[i], origin="lower", aspect="equal")
        plt.colorbar(orientation="horizontal")
        plt.axis("image")
        plt.title(r"$t=%.3f$" % t[i])
        plt.subplot(122)
        plt.imshow(
            np.gradient(phi[i], axis=0), origin="lower", aspect="equal")
        plt.colorbar(orientation="horizontal")
        plt.pause(0.05)


if __name__ == "__main__":
    plt.ion()
    # load_h5("droplet.eps_0.22.r0_3.5.turnover_1.2.h5")
    # os.chdir("128_128")
    # load_h5("droplet.eps_0.22.r0_3.5.turnover_1.2.h5")
    os.chdir(r"D:\data\droplet\droplet_2d")
    # load_h5("c_dct.eps_0.22.prolate_0.r0_4.turnover_1.yoff_35.h5")
    # os.chdir(r"..")
    load_h5(r"c_dct.eps_0.2.prolate_0.r0_3.5.turnover_0.6.x0_0.h5")
