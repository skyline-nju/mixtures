"""
Using semi-implicit Fourier-spectral method to solve Cahn-Hilliard equtions.

@author: skyline-nju
@data: 2018/5/19
@ref: L. Q. Chen & J. Shen 1998 Compu. Phys. Comm. Applications of
semi-implicit Fourier-spectral method to phase field equations
"""
import numpy as np
import matplotlib.pyplot as plt
import time


def f(c, c0_p, c0_m):
    return (c - c0_p) * (c - c0_m) * (c - c0_p + c - c0_m)


def get_freq(nx, ny, d, is_real=True):
    if is_real:
        kx = np.array([i / (Nx * spacing) for i in range(Nx // 2 + 1)])
    else:
        kx = np.zeros(nx)
        for i in range(nx):
            if i < nx // 2:
                kx[i] = i / (nx * spacing)
            else:
                kx[i] = (i - nx) / (nx * spacing)
    ky = np.zeros(ny)
    for i in range(ny):
        if i < ny // 2:
            ky[i] = i / (ny * spacing)
        else:
            ky[i] = (i - ny) / (ny * spacing)
    kx *= 2 * np.pi
    ky *= 2 * np.pi
    kxx, kyy = np.meshgrid(kx, ky)
    kk = kxx**2 + kyy**2
    return kx, ky, kk


class ExEulerFS:
    def __init__(self, nx, ny, d, dt, is_real=True):
        kx, ky, kk = get_freq(nx, ny, d, is_real)
        self.coeff_c = 1 - dt / 4 * kk * kk
        self.coeff_f = -dt * kk
        self.is_real = is_real

    def eval(self, c, c_k):
        f_c = f(c, c0_p, c0_m)
        if self.is_real:
            f_k = np.fft.rfft2(f_c)
            c_k = self.coeff_c * c_k + self.coeff_f * f_k
            c = np.fft.irfft2(c_k)
        else:
            f_k = np.fft.fft2(f_c)
            c_k = self.coeff_c * c_k + self.coeff_f * f_k
            c = np.fft.ifft2(c_k)
        return c, c_k


class SemiImFS:
    def __init__(self, nx, ny, d, dt, is_real=True):
        kx, ky, kk = get_freq(nx, ny, d, is_real)
        self.coeff1 = 1 / (1 + 0.25 * dt * kk * kk)
        self.coeff2 = -dt * kk
        self.coeff3 = 1 / (3 + 0.5 * dt * kk * kk)
        self.coeff4 = -2 * dt * kk
        self.is_real = is_real
        self.c = None
        self.c_k = None
        self.f_k = None

    def eval(self, c, c_k):
        f_c = f(c, c0_p, c0_m)
        if self.is_real:
            self.f_k = np.fft.rfft2(f_c)
            self.c_k = self.coeff1 * (c_k + self.coeff2 * self.f_k)
            self.c = np.fft.irfft2(c_k)
        else:
            self.f_k = np.fft.fft2(f_c)
            self.c_k = self.coeff1 * (c_k + self.coeff2 * self.f_k)
            self.c = np.fft.ifft2(c_k)
        return self.c, self.c_k

    def eval2(self, c, c_k):
        if self.c is None:
            return self.eval(c, c_k)
        else:
            if self.is_real:
                f_c = f(c, c0_p, c0_m)
                f_k = np.fft.rfft2(f_c)
                c_k_next = self.coeff3 * (4 * c_k - self.c_k + self.coeff4 *
                                          (2 * f_k - self.f_k))
                c_next = np.fft.irfft2(c_k_next)
            else:
                f_c = f(c, c0_p, c0_m)
                f_k = np.fft.fft2(f_c)
                c_k_next = self.coeff3 * (4 * c_k - self.c_k + self.coeff4 *
                                          (2 * f_k - self.f_k))
                c_next = np.fft.ifft2(c_k_next)
            self.c = c
            self.c_k = c_k
            self.f_k = f_k
            return c_next, c_k_next


dt = 0.5
spacing = 0.5
Nx = Ny = 128  # Nx and Ny should be even
Lx = Ly = Nx * spacing
np.random.seed(1)
c = np.ones((Ny, Nx)) * 0.5 + (np.random.rand(Ny, Nx) - 0.5) * 0.002
c0_p, c0_m = 0., 1.
is_real = True
if is_real:
    ck = np.fft.rfft2(c)
else:
    ck = np.fft.fft2(c)
# run = ExEulerFS(Nx, Ny, spacing, dt, is_real)
run = SemiImFS(Nx, Ny, spacing, dt, is_real)

plt.ion()
plt.imshow(c, origin="lower", vmin=0, vmax=1)
plt.colorbar()
plt.pause(0.1)

t_beg = time.time()
for i in range(10000):
    if i != 0 and i % 1000 == 0:
        plt.clf()
        plt.imshow(c.real, origin="lower", vmin=0, vmax=1)
        plt.colorbar()
        plt.title(r"$t=%g$" % (i * dt))
        plt.pause(0.1)
    c, ck = run.eval2(c, ck)
t_end = time.time()
print("totally cost", t_end - t_beg)
plt.close()
