import numpy as np
import scipy.special as sp
import scipy.interpolate as interp
import matplotlib.pyplot as plt
import scipy.optimize as op
import matplotlib

matplotlib.rcParams.update({'font.size': 15})

HBAR = 1.0545718E-34
ME = 9.10938356E-31
E0 = 1.60217662E-19
NE = 5.541318196761572e28
NE = 5.476415562682574e28

mu0 = 1.2566370614E-06;
ep0 = 8.85418781762E-12;
h = 4.135667516000000e-15;
c = 3e8;


def x_Bessel_prime(n, x, j):
    xj_prime = x * j[:, 1] - n * j[:, 0];
    return xj_prime;

def Mie_Solver(Sphere_Radius, Sphere_refractive_index, Wavelengths, RefractiveIndex_Outside = 1, max_iter = 20):
    c = 3e8;
    N_m = RefractiveIndex_Outside;  # vacuum
    Lambda = Wavelengths
    Freq = c / Lambda;
    omega = 2 * np.pi * Freq;

    k = 2 * np.pi * N_m / Lambda;
    x = Sphere_Radius * k;

    if hasattr(Sphere_refractive_index, "__len__"):
        N_s = Sphere_refractive_index
    else:
        N_s = Wavelengths / Wavelengths * Sphere_refractive_index

    N = N_s / N_m;

    j_mx = np.zeros([len(omega), 3], dtype='complex128');
    j_x = np.zeros([len(omega), 3], dtype='complex128');
    y_x = np.zeros([len(omega), 3], dtype='complex128');
    xjx_prime = np.zeros([len(omega), 3], dtype='complex128');
    NxjNx_prime = np.zeros([len(omega), 3], dtype='complex128');
    h_x = np.zeros([len(omega), 3], dtype='complex128');
    xhx_prime = np.zeros([len(omega), 3], dtype='complex128');

    I = 1j

    i = 0;
    j_x[:, 0] = sp.jn(i + 0.5, x) * np.sqrt(np.pi / (2 * x))
    j_mx[:, 0] = sp.jn(i + 0.5, N * x) * np.sqrt(np.pi / (2 * N * x))
    y_x[:, 0] = sp.yv(i + 0.5, x) * np.sqrt(np.pi / (2 * x))
    h_x[:, 0] = j_x[:, 0] + I * y_x[:, 0];

    Q_sca = 0;
    Q_ext = 0;

    A_n_track = []
    B_n_track = []

    for i in range(1, max_iter + 1):
        # print(i)

        j_x[:, 2] = j_x[:, 1];
        j_x[:, 1] = j_x[:, 0];

        j_mx[:, 2] = j_mx[:, 1];
        j_mx[:, 1] = j_mx[:, 0];

        y_x[:, 2] = y_x[:, 1];
        y_x[:, 1] = y_x[:, 0];

        j_x[:, 0] = sp.jv(i + 0.5, x) * np.sqrt(np.pi / (2 * x))
        j_mx[:, 0] = sp.jv(i + 0.5, N * x) * np.sqrt(np.pi / (2 * N * x))
        y_x[:, 0] = sp.yv(i + 0.5, x) * np.sqrt(np.pi / (2 * x))

        h_x[:, 2] = h_x[:, 1];
        h_x[:, 1] = h_x[:, 0];
        h_x[:, 0] = j_x[:, 0] + I * y_x[:, 0];

        xjx_prime[:, 0] = x_Bessel_prime(i, x, j_x);
        NxjNx_prime[:, 0] = x_Bessel_prime(i, N * x, j_mx);
        xhx_prime[:, 0] = x_Bessel_prime(i, x, h_x);

        A_n = N * N * j_mx[:, 0] * xjx_prime[:, 0] - j_x[:, 0] * NxjNx_prime[:, 0];
        A_n = - A_n / (N * N * j_mx[:, 0] * xhx_prime[:, 0] - h_x[:, 0] * NxjNx_prime[:, 0]);

        B_n = j_mx[:, 0] * xjx_prime[:, 0] - j_x[:, 0] * NxjNx_prime[:, 0];
        B_n = - B_n / (j_mx[:, 0] * xhx_prime[:, 0] - h_x[:, 0] * NxjNx_prime[:, 0]);

        A_n_track.append(A_n)
        B_n_track.append(B_n)

        Q_sca = Q_sca + (2 * i + 1) * (np.abs(A_n) * np.abs(A_n) + np.abs(B_n) * np.abs(B_n));
        Q_ext = Q_ext + (2 * i + 1) * (np.real(A_n + B_n));

    Q_sca = Q_sca * 2 / (x * x);
    Q_ext = -Q_ext * 2 / (x * x);
    Q_abs = Q_ext - Q_sca;

    return Q_sca, Q_abs, Q_ext
