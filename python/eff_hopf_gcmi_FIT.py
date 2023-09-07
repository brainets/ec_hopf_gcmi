from numpy import matlib
import numpy as np
import scipy.io as sio
from scipy.signal import butter, filtfilt, detrend
from scipy.stats import pearsonr
from scipy.special import psi, erfcinv
from scipy.linalg import cholesky
from frites.core import gcmi_nd_cc
import pdb

# specify condition and reaction time
# condition=1; % 1:2 easy, hard
# reaction=1;  % 1:3 fast, middle, slow


def demean(x, dim=None):
    if dim is None:
        dim = 0 if x.shape[0] > 1 else 1 if x.shape[1] > 1 else 0

    dimsize = x.shape[dim]
    dimrep = np.ones(len(x.shape), dtype=int)
    dimrep[dim] = dimsize

    x = x - np.tile(np.mean(x, axis=dim, keepdims=True), dimrep)

    return x


def detrend_and_demean(signal):
    # Detrend by fitting a linear trend and subtracting it
    x = np.arange(len(signal))
    trend = np.polyfit(x, signal, 1)
    detrended_signal = signal - np.polyval(trend, x)

    # Demean the detrended signal
    demeaned_signal = demean(detrended_signal)

    return demeaned_signal


def eff_hopf_gcmi_FIT(reaction, condition, simulation=None):
    print("######## starting Hopf model inversion ###########")
    np.random.seed(None)  # Necessary to generate random numbers in Python

    # Load empirical data calculated with FIT
    data = sio.loadmat('../data/fit_difficulty_bis.mat')
    # print(data)
    signal_fit = data['fit_matrix']
    fitm = np.squeeze(np.nanmean(signal_fit[:, :, :], axis=2))
    fitm[np.isnan(fitm)] = 0

    # Load SEEG Gaussian-Copula Mutual Information, empirical data
    data = sio.loadmat('../data/dfc-rt-binned_estimator-gcmi_freq-f50f150_align-sample_stim_lowpass-1.mat')
    FC = np.squeeze(data['data'][:, :, condition - 1, reaction - 1])

    # FIT matrix
    C = fitm
    C = C - np.diag(C)  # is this right? on matlab: C = C - diag(diag(C));
    C = C / np.max(C) * 0.2

    FCemp = FC
    # nmask3 = ~np.isnan(FCemp)  # Mask of NaNs
    # nmask2 = np.logical_not(np.isnan(FCemp))
    nmask = np.ones((82, 82))
    np.fill_diagonal(nmask, 0)  # put diagonal to zero
    print("nmask ", nmask)
    NSIM = 1  # Total simulations

    # Parameters of the data
    TR = 1  # Repetition Time (seconds)
    N = 82  # Total regions
    Isubdiag = np.tril_indices(N, -1)
    f_diff = 0.05 * np.ones(N)  # Frequency

    # Bandpass filter settings
    fnq = 1 / (2 * TR)  # Nyquist frequency
    flp = 0.008  # Lowpass frequency of filter (Hz)
    fhi = 0.08  # Highpass frequency
    Wn = [flp / fnq, fhi / fnq]  # Butterworth bandpass non-dimensional frequency
    k = 2  # 2nd order butterworth filter
    bfilt, afilt = butter(k, Wn, 'bandpass')  # Construct the filter

    # Parameters HOPF
    a = -0.02 * np.ones((N, 2))
    omega = np.repeat(2 * np.pi * f_diff[:, None], 2, axis=1)
    omega[:, 0] = -omega[:, 0]
    dt = 0.1 * TR / 2

    sig = 0.01
    dsig = np.sqrt(dt) * sig
    Tmax = 10000

    GCsim2 = np.zeros((NSIM, N, N))  # Simulated Gaussian-Copula Mutual Information
    Cnew = C.copy()  # Effective connectivity matrix updated in each iteration

    # initialize
    fcsimuli = np.zeros((500, N, N))
    fittFC = np.zeros(500)
    corr2FCSC = np.zeros(500)
    corr2FCemp_sim = np.zeros(500)
    iEffectiveConnectivity = np.zeros((500, N, N))

    # for iteration in range(500):
    for iteration in range(2):
        print("Iteration:", iteration)
        wC = Cnew  # Updated in each iteration
        sumC = np.tile(np.sum(wC, axis=1), (2, 1)).T  # For sum Cij * xj

        for sub in range(NSIM):
            xs = np.zeros((Tmax, N))
            z = 0.1 * np.ones((N, 2))  # --> x = z[:, 0], y = z[:, 1]
            nn = 0

            # Discard first 3000 time steps
            for t in np.arange(0, 3000 + dt, dt):
                suma = wC.dot(z) - sumC * z  # ai
                zz = z[:, ::-1]  # Flipped z because (x * x + y * y)
                z = z + dt * (a * z + zz * omega - z * (z * z + zz * zz) + suma) + (dsig * np.random.randn(N, 2))  # ?

            # Actual modeling (x = simulated_signal (Interpretation), y = some other oscillation)
            for t in np.arange(0, (Tmax - 1) * TR + dt, dt):
                suma = wC.dot(z) - sumC * z  # ai
                zz = z[:, ::-1]  # Flipped z because (x * x + y * y)
                z = z + dt * (a * z + zz * omega - z * (z * z + zz * zz) + suma) + (dsig * np.random.randn(N, 2))
                if np.abs(np.mod(t, TR)) < 0.01:
                    xs[nn, :, np.newaxis] = np.atleast_2d(z[:, 0]).T.conj()  # better solution instead of np.newaxis?
                    nn = nn + 1


            simulated_signal = np.atleast_2d(xs).T.conj()
            signal_filt22 = np.zeros((N, nn))
            for seed in range(N):
                demeaned_signal = detrend_and_demean(simulated_signal[seed, :])
                signal_filt22[seed, :] = filtfilt(bfilt, afilt, demeaned_signal)

            signal_filt = signal_filt22[:, 20:-20]  # not used

            for i in range(N):
                for j in range(N):
                    if nmask[i, j]:
                        p = signal_filt22[i, :, np.newaxis]
                        o = signal_filt22[j, :, np.newaxis]
                        # import pdb;pdb.set_trace()
                        # print("are the arrays equal? ", np.array_equal(p, o))
                        GCsim2[sub, i, j] = gcmi_nd_cc(np.atleast_2d(p).T.conj(), np.atleast_2d(o).T.conj())
                    else:
                        GCsim2[sub, i, j] = 0

        fcsimul = np.mean(GCsim2, axis=0)

        fcsimuli[iteration, :, :] = fcsimul - np.diag(np.diag(fcsimul))

        fittFC[iteration] = np.sqrt(np.nanmean((FCemp[Isubdiag] - fcsimul[Isubdiag]) ** 2))

        corr2FCSC[iteration] = np.corrcoef(fcsimul[Isubdiag], C[Isubdiag])[0, 1]

        fc = FCemp[Isubdiag]
        idx = np.isnan(fc)
        fc = fc[~idx]
        fcsimdia = fcsimul[Isubdiag]
        fcsimdia = fcsimdia[~idx]
        corr2FCemp_sim[iteration] = np.corrcoef(fcsimdia, fc)[0, 1]

        for i in range(N):
            for j in range(N):
                if nmask[i, j]:
                    if C[i, j] > 0 or j == N // 2 + i:
                        Cnew[i, j] = Cnew[i, j] + 0.005 * (FCemp[i, j] - fcsimul[i, j])
                        if Cnew[i, j] < 0 or np.isnan(Cnew[i, j]):
                            Cnew[i, j] = 0

        Cnew = Cnew / np.max(Cnew) * 0.2
        iEffectiveConnectivity[iteration, :, :] = Cnew
