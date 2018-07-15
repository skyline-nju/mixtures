import numpy as np
from mayavi import mlab
import os


def load_h5(filename):
    import h5py
    f = h5py.File(filename, "r")
    phi = f['1']['Phi']
    t = f['1']['t']
    # X, Y, Z = np.mgrid[-30:30:120j, -30:30:120j, -30:30:120j]
    X, Y, Z = np.mgrid[-60:60:240j, -60:60:240j, -60:60:240j]
    # X, Y, Z = np.mgrid[-48:48:192j, -30:30:120j, -30:30:120j]
    for i in range(t.size):
        mlab.contour3d(X, Y, Z, phi[i], contours=[0.5], transparent=True)
        # mlab.pipeline.image_plane_widget(
        #     mlab.pipeline.scalar_field(phi[i]),
        #     plane_orientation="z_axes",
        #     slice_index=60)
        # mlab.pipeline.image_plane_widget(
        #     mlab.pipeline.scalar_field(phi[i]),
        #     plane_orientation="x_axes",
        #     slice_index=60)
        # mlab.pipeline.image_plane_widget(
        #     mlab.pipeline.scalar_field(phi[i]),
        #     plane_orientation="y_axes",
        #     slice_index=96)
        mlab.axes(xlabel="x", ylabel="y", zlabel="z")
        mlab.outline()
        mlab.show()
    mlab.close(all=True)


def test_contour3d():
    x, y, z = np.mgrid[-5:5:64j, -5:5:64j, -5:5:64j]

    scalars = x * x + y * y + z * z
    obj = mlab.contour3d(x, y, z, scalars, contours=[1.], transparent=True)
    mlab.axes(xlabel="x", ylabel="y", zlabel="z")
    return obj


if __name__ == "__main__":
    os.chdir(r"D:\data\droplet\droplet_3d\large")
    load_h5("dct3d.eps_0.2.prolate_0.r0_5.turnover_1.h5")
    # obj = test_contour3d()

    # mlab.show()
