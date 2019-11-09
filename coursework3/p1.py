"""M345SC Homework 3, part 1
Tudor Trita Trita
CID: 01199397
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.signal import hann
from scipy.linalg import solve_banded
from scipy.sparse import diags


def nwave(alpha, beta, Nx=256, Nt=801, T=200, display=False):
    """
    Question 1.1
    Simulate nonlinear wave model

    Input:
    alpha, beta: complex model parameters
    Nx: Number of grid points in x
    Nt: Number of time steps
    T: Timespan for simulation is [0,T]
    Display: Function creates contour plot of |g| when true

    Output:
    g: Complex Nt x Nx array containing solution
    """

    # generate grid
    L = 100
    x = np.linspace(0, L, Nx + 1)
    x = x[:-1]

    def RHS(f, t, alpha, beta):
        """Computes dg/dt for model eqn.,
        f[:N] = Real(g), f[N:] = Imag(g)
        Called by odeint below
        """
        g = f[:Nx] + 1j * f[Nx:]

        # -----------
        c = np.fft.fft(g) / Nx
        n = np.fft.fftshift(np.arange(-Nx / 2, Nx / 2))
        k = 2 * np.pi * n / L
        d2g = Nx * np.fft.ifft(- k**2 * c)

        # -----------
        dgdt = alpha * d2g + g - beta * g * g * g.conj()
        df = np.zeros(2 * Nx)
        df[:Nx] = dgdt.real
        df[Nx:] = dgdt.imag
        return df

    # set initial condition
    g0 = np.random.rand(Nx) * 0.1 * hann(Nx)
    f0 = np.zeros(2 * Nx)
    f0[:Nx] = g0
    t = np.linspace(0, T, Nt)

    # compute solution
    f = odeint(RHS, f0, t, args=(alpha, beta))
    g = f[:, :Nx] + 1j * f[:, Nx:]

    if display:
        plt.figure()
        plt.contour(x, t, g.real)
        plt.xlabel('x')
        plt.ylabel('t')
        plt.title('Contours of Real(g)')
    return g


def analyze(display=(False, False, False)):
    """
    Question 1.2
    Input: display: Boolean tuple of size 5. Set display[i] to True to compute
                    and show Figure i. display[0] computes figs 21 and 22.
    Output: None

    FIGURES CREATED:
    2.1  Plot of energies for different Fourier Coeff. at time T = 50
    2.2  Plot of energies for different Fourier Coeff. at time T = 250
    2.3  Plot of energies for different Fourier Coeff varying Nx for case A
    2.4  Plot of energies for different Fourier Coeff varying Nx for case B
    2.5  Plot of energies for different Fourier Coeff varying Nt for case A
    2.6  Plot of energies for different Fourier Coeff varying Nt for case B

    Discussion:
    Here energy refers to the amplitude of the corresponding Fourier Coefficients.

    Figures 2.1 and 2.2:
    These figures show how the energies of the Fourier coefficients change
    when moving from T = 50 to T = 250. The purpose of these plots is to
    investigate how the energies change the further the wave has progressed.
    We can see that for T = 50, the energies for Case B are much lower than
    for Case A. Also, the energies for Case A are a lot more disordered than
    Case B. This suggests that Case A is the more chaotic one. For Case B,
    the positive coefficients are more chaotic than the negative ones. For
    T = 250, the energies are more chaotic for each coeff., and the overall
    amount of energy seems to have increased for each N.

    Figures 2.3 and 2.4:
    These figures show what happens to the energies when we vary Nx. We can see
    that if Nx is smaller, we lose coefficients with lower energies. This
    suggests that if the grid is larger, there is more sensitivity to
    coefficients with smaller energies. Again, there seems to be more
    variability in energies for Case A, however, Case B has ne coefficient
    with the highest energy at Nx = 100.

    Figures 2.5 and 2.6:
    These figures show how the energies of the Fourier coefficients change
    when varying Nt. We can see that there seems to be not a lot of change
    between Cases A and B, thus implying that Nt is not major factor in
    energies of Fourier Coefficients.
    """
    def analysis_time(ctr, T):
        tdiff = T * 4
        x = np.linspace(0, L, Nx + 1)
        x = x[:-1]
        c1 = np.fft.fft(g1[tdiff, :]) / Nx
        c2 = np.fft.fft(g2[tdiff, :]) / Nx
        n = np.arange(-Nx / 2, Nx / 2)
        plt.figure(figsize=(10, 7))
        plt.semilogy(n, np.abs(np.fft.fftshift(c1))**2, 'bx', label="A")
        plt.semilogy(n, np.abs(np.fft.fftshift(c2))**2, 'r^', label="B")
        plt.xlabel("n")
        plt.ylabel("Energy")
        plt.legend()
        plt.title('Name: Tudor Trita Trita, CID:01199397 \n Figure 2.' + str(ctr) + ', ' +
                  'Function: analyze \n Plot of energies vs Fourier coeff. ' +
                  'for Cases A and B when T = ' + str(T))
        # plt.savefig('fig2' + str(ctr) + '.png')
        plt.show()
        return None

    # Fixed parameters:
    L = 100

    # Case 1:
    alpha1 = 1 - 2j
    beta = 1 + 2j
    # Case 2:
    alpha2 = 1 - 1j

    # Figures 1-2: Varying T
    if display[0]:
        Nx, Nt, T = 256, 801*4, 300
        g1 = nwave(alpha1, beta, Nx=Nx, Nt=Nt, T=T)
        g2 = nwave(alpha2, beta, Nx=Nx, Nt=Nt, T=T)
        Tarray = [50, 250]
        for i, j in enumerate(Tarray):
            analysis_time(i + 1, j)

    # Figure 3-4: Comparing what happens if we switch Nx
    if display[1]:
        Nx, Nt, T = 256, 801, 100
        g1 = nwave(alpha1, beta, Nx=Nx, Nt=Nt, T=T)
        g2 = nwave(alpha2, beta, Nx=Nx, Nt=Nt, T=T)
        n = np.arange(-Nx / 2, Nx / 2)
        g1_fft = np.fft.fft(g1[50:, :])/Nx
        g2_fft = np.fft.fft(g1[50:, :])/Nx

        Nx = 128
        g1small = nwave(alpha1, beta, Nx=Nx, Nt=Nt, T=T)
        g12_fft = np.fft.fft(g1small[50:, :])/Nx
        n1 = np.arange(-Nx/2, Nx/2)

        plt.figure(figsize=(10, 7))
        plt.semilogy(n1, np.fft.fftshift(np.abs(g12_fft[-1, :])), 'r^')
        plt.semilogy(n, np.fft.fftshift(np.abs(g1_fft[-1, :])), 'bx')
        plt.legend(('Nx=100', 'Nx=256'))
        plt.xlabel('n')
        plt.ylabel('Energy')
        plt.title('Name: Tudor Trita Trita, CID:01199397 \n Figure 2.3 ' +
                  'Function: analyze \n Plot of energies vs Fourier coeff. ' +
                  'for Case A when T = 100 varying Nx')
        # plt.savefig('fig23.png')
        plt.show()

        g2small = nwave(alpha2, beta, Nx=Nx, Nt=Nt, T=T)  # Case B
        g22_fft = np.fft.fft(g2small[50:, :])/Nx
        n2 = np.arange(-Nx/2, Nx/2)

        plt.figure(figsize=(10, 7))
        plt.semilogy(n2, np.fft.fftshift(np.abs(g22_fft[-1, ])), 'r^')
        plt.semilogy(n, np.fft.fftshift(np.abs(g2_fft[-1, ])), 'bx')
        plt.legend(('Nx=100', 'Nx=256'))
        plt.xlabel('n')
        plt.ylabel('Energy')
        plt.title('Name: Tudor Trita Trita, CID:01199397 \n Figure 2.4 ' +
                  'Function: analyze \n Plot of energies vs Fourier coeff. ' +
                  'for Case B when T = 100 varying Nx')
        # plt.savefig('fig24.png')
        plt.show()

    # Figure 5-6: Comparing what happens if we switch Nt
    if display[2]:
        Nx, Nt, T = 256, 801, 100
        g1 = nwave(alpha1, beta, Nx=Nx, Nt=Nt, T=T)
        g2 = nwave(alpha2, beta, Nx=Nx, Nt=Nt, T=T)
        n = np.arange(-Nx / 2, Nx / 2)
        g1_fft = np.fft.fft(g1[50:, :])/Nx
        g2_fft = np.fft.fft(g1[50:, :])/Nx

        Nt = 150
        g1small = nwave(alpha1, beta, Nx=Nx, Nt=Nt, T=T)
        g12_fft = np.fft.fft(g1small[50:, :])/Nx
        n1 = np.arange(-Nx/2, Nx/2)

        plt.figure(figsize=(10, 7))
        plt.semilogy(n1, np.fft.fftshift(np.abs(g12_fft[-1, :])), 'r^')
        plt.semilogy(n, np.fft.fftshift(np.abs(g1_fft[-1, :])), 'bx')
        plt.legend(('Nt=150', 'Nt=801'))
        plt.xlabel('n')
        plt.ylabel('Energy')
        plt.title('Name: Tudor Trita Trita, CID:01199397 \n Figure 2.5 ' +
                  'Function: analyze \n Plot of energies vs Fourier coeff. ' +
                  'for Case A when T = 100 varying Nt')
        # plt.savefig('fig25.png')
        plt.show()

        g2small = nwave(alpha2, beta, Nx=Nx, Nt=Nt, T=T)
        g22_fft = np.fft.fft(g2small[50:, :])/Nx
        n2 = np.arange(-Nx/2, Nx/2)

        plt.figure(figsize=(10, 7))
        plt.semilogy(n2, np.fft.fftshift(np.abs(g22_fft[-1, ])), 'r^')
        plt.semilogy(n, np.fft.fftshift(np.abs(g2_fft[-1, ])), 'bx')
        plt.legend(('Nt=150', 'Nt=801'))
        plt.xlabel('n')
        plt.ylabel('Energy')
        plt.title('Name: Tudor Trita Trita, CID:01199397 \n Figure 2.6 ' +
                  'Function: analyze \n Plot of energies vs Fourier coeff. ' +
                  'for Case B when T = 100 varying Nt')
        # plt.savefig('fig26.png')
        plt.show()

    return None


def wavediff(display=(False, False, False, False)):
    """
    Question 1.3
    Input: display: Boolean tuple of size 4. Set display[i] to True to compute
                    and show Figure i.
    Output: None

    FIGURES CREATED:
    3.1 Varying Nx and seeing absolute value of difference between methods.
    3.2 Loglog plot of wall-times comparing both method's speed.
    3.3 Magnitudes of the derivatives obtained from both methods for low Nx.
    3.4 Magnitudes of the derivatives obtained from both methods for high Nx.

    Discussion:

    When comparing both methods, there are two fundamental factors that we
    look at; accuracy and speed.

    Accuracy:

    We first turn to the accuracy of the two methods. For this we will compare
    the dependency of the accuracy of the methods on Nx. We can see that as Nx
    increases, both methods's results seem to converge. Looking at Figure 3.1,
    we see thatfor Nx = 256, the mean absolute value of the difference in
    derivative (m.a.v.d.d.) is around 10e-6. The variance of the absolute value
    of the differece in derivative (v.a.v.d.d.). For Nx = 512, the m.a.v.d.d.
    increases and the ab.v.diff is around 10-9. The v.a.v.d.d. is slightly
    higher than for Nx = 256. For Nx = 1024, the m.a.v.d.d is similar to the
    case Nx = 512, but the v.a.v.d.d. is a lot lower. This suggests, that FD
    has reached its limit in accuracy, and we can say that increasing Nx further
    would only increase coputational cost but no increase in accuracy would be
    gained. Thus by using DFT as a reference point, it is clear that the
    accuracy of FD increases as Nx increases up to a limit.

    However there is a big difference when considering the boundaries. The
    accuracy of the FD method is a lot lower for the boundaries than for the
    central section of Nx, thus suggesting that the one-sided FD method is not
    as accurate compared with the central 4th order FD method.

    We now turn to Figures 3.2 and 3.3:
    In Fig 3.2, low Nx, with DFT as a reference guide, we see that the FD method is
    relatively accurate for portions of the derivative where there is not much
    change. When there are peaks and troughs in the magnitude of the derivatives,
    the two methods do not agree in value. However, when Nx is increased as seen
    in Fig 3.3, when Nx = 512, the FD method seems to agree with the FTD method
    well even in sharp troughs and changes in derivative

    Speed:

    Now we turn to the speed of both methods. Figure 3.2 clearly shows that
    the wall-time for the DFT method is faster than the DF. In fact, on average,
    the wall-time for the DFT method is approximately 4 times faster. Both methods
    exhibit an exponential increase in time as Nx increases, as shown by the
    straight line on the log-log plot. This is unsurprising as both algorithms
    are O(Nx^2).

    One advantage of the FD method is that it works for general waves, whereas
    the FDT in it's current form works for periodic waves only. For DFT to
    work on non-periodic waves, we would need to use a windowing method, and
    this will add complexity and thus time to it's execution.
    """
    def DFT(g, Nx=256):
        """ Function to perform Fourier-based
        calculations of derivative. """
        c = np.fft.fft(g) / Nx
        n = np.fft.fftshift(np.arange(-Nx / 2, Nx / 2))
        k = 2 * np.pi * n / L
        dg = Nx * np.fft.ifft(1j * k * c)
        return dg

    def FD(g, Nx=256):
        """ Function to perform Finite Difference
        calculations of derivative of g."""

        # LHS
        Ab = np.array([np.ones(Nx)*alpha1, np.ones(Nx), np.ones(Nx)*alpha1])
        Ab[0, 0], Ab[0, 1], Ab[-1, -1], Ab[-1, -2] = 0, alpha2, 0, alpha2

        # RHS
        ones = np.ones(Nx)
        z_gv = np.zeros(Nx)
        z_gv[0], z_gv[-1] = a2 * hinv, -a2 * hinv

        # Bottom vecs of RHS
        # Fourth order FD:
        l_agv = -a1 * ones[1:] * hinv2
        l_bgv = -b1 * ones[2:] * hinv4
        l_cgv = -c1 * ones[3:] * hinv6

        # One-sided method bottom:
        l_agv[-1], l_bgv[-1], l_cgv[-1] = -b2 * hinv, -c2 * hinv, -d2 * hinv

        # Top vecs of RHS
        # Fourth order FD:
        r_agv = a1 * ones[1:] * hinv2
        r_bgv = b1 * ones[2:] * hinv4
        r_cgv = c1 * ones[3:] * hinv6

        # One-sided method top:
        r_agv[0], r_bgv[0], r_cgv[0] = b2 * hinv, c2 * hinv, d2 * hinv

        b = diags([[b1*hinv4, 0], [c1*hinv6, c1*hinv6, 0], l_cgv, l_bgv, l_agv,
                   z_gv, r_agv, r_bgv, r_cgv, [0, -c1*hinv6, -c1*hinv6],
                   [0, -b1*hinv4]], [-Nx + 2, -Nx + 3, -3, -2, -1,
                                     0, 1, 2, 3, Nx - 3, Nx - 2])

        B = b @ g
        dg = solve_banded((1, 1), Ab, B)
        return dg

    # Initialising parameters:
    L = 100
    Nt = 801
    T = 100

    # Parameters for FD
    alpha1 = 3 / 8
    a1 = 25 / 16
    b1 = 1 / 5
    c1 = -1 / 80
    alpha2 = 3
    a2 = -17 / 6
    b2 = 3 / 2
    c2 = 3 / 2
    d2 = -1 / 6

    if display[0]:
        colours = ['b--', 'r--', 'g--']
        Nxarray = [256, 512, 1024]

        g0 = nwave(1 - 1j, 1 + 2j, Nx=Nxarray[0], Nt=Nt, T=T)[800]
        g1 = nwave(1 - 1j, 1 + 2j, Nx=Nxarray[1], Nt=Nt, T=T)[800]
        g2 = nwave(1 - 1j, 1 + 2j, Nx=Nxarray[2], Nt=Nt, T=T)[800]
        Glist = [g0, g1, g2]

        # FIGURE 31: Plot of errors
        plt.figure(figsize=(10, 7))
        for i, j in enumerate(Nxarray):
            # Defining h
            x = np.linspace(0, L, j + 1)
            x = x[:-1]
            hinv = 1 / (x[1] - x[0])

            # Defining h inverses
            hinv2 = hinv / 2
            hinv4 = hinv / 4
            hinv6 = hinv / 6

            dg1 = DFT(Glist[i], Nx=j)
            dg2 = FD(Glist[i], Nx=j)
            plt.semilogy(x, np.abs(dg1 - dg2), colours[i],
                         label=r'$Nx = $' + str(j))
        plt.xlabel(r'$x$')
        plt.ylabel(r'$|dg_1 - dg_2|$')
        plt.legend()
        plt.grid()
        plt.title('Name: Tudor Trita Trita, CID:01199397 \n Figure 3.1, ' +
                  'Function: wavediff \n Plot of absolute difference of ' +
                  'derivative for FTD and FD methods')
        # plt.savefig('fig31.png')
        plt.show()

    # FIGURE 32: Plot of Wall-times:
    iters1 = 100
    # Sort out issue of calculating g twice
    Nxarray = [8, 16, 32, 64, 128, 256, 512, 1024]
    reps1 = len(Nxarray)
    tarray1 = np.zeros(reps1)
    tarray2 = np.zeros(reps1)

    if display[1]:
        for i, j in enumerate(Nxarray):
            # Defining h
            x = np.linspace(0, L, j + 1)
            x = x[:-1]
            h = x[1] - x[0]

            # Defining h inverses
            hinv = 1 / h
            hinv2 = 1 / (2 * h)
            hinv4 = 1 / (4 * h)
            hinv6 = 1 / (6 * h)

            gin = nwave(1 - 1j, 1 + 2j, Nx=Nxarray[i], Nt=Nt, T=T)[800]

            for k in range(iters1):
                t1 = time.perf_counter()
                DFT(gin, Nx=j)
                t2 = time.perf_counter()
                tarray1[i] += t2 - t1

                t1 = time.perf_counter()
                FD(gin, Nx=j)
                t2 = time.perf_counter()
                tarray2[i] += t2 - t1
            tarray1 /= iters1
            tarray2 /= iters1

        plt.figure(figsize=(10, 7))
        plt.loglog(Nxarray, tarray1, 'bx--', label='Walltime - DFT')
        plt.loglog(Nxarray, tarray2, 'rx--', label='Walltime - FD')
        plt.legend()
        plt.xlabel(r'$Nx$')
        plt.ylabel('Wall-times')
        plt.grid()
        plt.title('Name: Tudor Trita Trita, CID:01199397 \n Figure 32, ' +
                  'Function: wavediff \n Loglog plot of walltimes for both ' +
                  'FTD and FD methods varying Nx.')
        # plt.savefig('fig32.png')
        plt.show()

    if display[2]:
        Nx = 80
        gin = nwave(1 - 1j, 1 + 2j, Nx=Nx, Nt=Nt, T=T)[800]

        # Defining h
        x = np.linspace(0, L, Nx + 1)
        x = x[:-1]
        h = x[1] - x[0]

        # Defining h inverses
        hinv = 1 / h
        hinv2 = 1 / (2 * h)
        hinv4 = 1 / (4 * h)
        hinv6 = 1 / (6 * h)

        dg1 = DFT(gin, Nx=Nx)
        dg2 = FD(gin, Nx=Nx)

        plt.figure(figsize=(13, 8))
        plt.semilogy(x, np.abs(dg1), 'kx', label='DFT')
        plt.semilogy(x, np.abs(dg2), 'b--', label='FD')
        plt.legend()
        plt.xlabel('x')
        plt.grid()
        plt.ylabel('Magnitude of Derivatives')
        plt.title('Name: Tudor Trita Trita, CID:01199397 \n Figure 33, ' +
                  'Function: wavediff \n Plot of magnitudes of derivative ' +
                  'comparison between FTD and FD methods for Nx = ' + str(Nx))
        # plt.savefig('fig33.png')
        plt.show()

    if display[3]:
        Nx = 512
        gin = nwave(1 - 1j, 1 + 2j, Nx=Nx, Nt=Nt, T=T)[800]

        # Defining h
        x = np.linspace(0, L, Nx + 1)
        x = x[:-1]
        h = x[1] - x[0]

        # Defining h inverses
        hinv = 1 / h
        hinv2 = 1 / (2 * h)
        hinv4 = 1 / (4 * h)
        hinv6 = 1 / (6 * h)

        dg1 = DFT(gin, Nx=Nx)
        dg2 = FD(gin, Nx=Nx)

        plt.figure(figsize=(13, 8))
        plt.semilogy(x, np.abs(dg1), 'kx', label='DFT')
        plt.semilogy(x, np.abs(dg2), 'b--', label='FD')
        plt.legend()
        plt.xlabel('x')
        plt.grid()
        plt.ylabel('Magnitude of Derivatives')
        plt.title('Name: Tudor Trita Trita, CID:01199397 \n Figure 34, ' +
                  'Function: wavediff \n Plot of magnitudes of derivative ' +
                  'comparison between FTD and FD methods for Nx = ' + str(Nx))
        # plt.savefig('fig34.png')
        plt.show()
    return None


if __name__ == '__main__':
    display1 = (False, False, False)  # Toggle
    analyze(display=display1)

    display2 = (False, False, False, False)  # Toggle
    wavediff(display=display2)
