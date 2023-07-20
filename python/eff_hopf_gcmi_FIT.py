import numpy as np
import scipy.io as sio
from scipy.signal import butter, filtfilt
from scipy.stats import pearsonr
from frites.conn import gcmi_nd_cc

def eff_hopf_gcmi_FIT(simulation, reaction, condition):
    np.random.seed(None)  # Necessary to generate random numbers in Python

    # Load empirical data calculated with FIT
    data = sio.loadmat('../data/fit_difficulty_bis.mat')
    signal_fit = data['fit_matrix']
    fitm = np.nanmean(signal_fit, axis=2)
    fitm[np.isnan(fitm)] = 0

    # Load SEEG Gaussian-Copula Mutual Information, empirical data
    data = sio.loadmat('../data/dfc-rt-binned_estimator-gcmi_freq-f50f150_align-sample_stim_lowpass-1.mat')
    FC = data['data'][:, :, condition-1, reaction-1]

    # FIT matrix
    C = fitm
    np.fill_diagonal(C, 0)
    C = C / np.max(C) * 0.2

    FCemp = FC
    nmask = ~np.isnan(FCemp)  # Mask of NaNs
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
    bfilt, afilt = butter(k, Wn)  # Construct the filter

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

    for iteration in range(500):
        print("Iteration:", iteration)
        wC = Cnew  # Updated in each iteration
        sumC = np.tile(np.sum(wC, axis=1), (2, 1)).T  # For sum Cij * xj

        for sub in range(NSIM):
            xs = np.zeros((Tmax, N))
            z = 0.1 * np.ones((N, 2))  # --> x = z[:, 0], y = z[:, 1]
            nn = 0

            # Discard first 3000 time steps
            for t in np.arange(0, 3000 + dt, dt):
                suma = wC * z - sumC * z  # sum(Cij * xi) - sum(Cij) * xj
                zz = z[:, ::-1]  # Flipped z because (x * x + y * y)
                z = z + dt * (a * z + zz * omega - z * (z * z + zz * zz) + suma) + dsig * np.random.randn(N, 2)

            # Actual modeling (x = simulated_signal (Interpretation), y = some other oscillation)
            for t in np.arange(0, (Tmax - 1) * TR + dt, dt):
                suma = wC * z - sumC * z  # sum(Cij * xi) - sum(Cij) * xj
                zz = z[:, ::-1]  # Flipped z because (x * x + y * y)
                z = z + dt * (a * z + zz * omega - z * (z * z + zz * zz) + suma) + dsig * np.random.randn(N, 2)
                if np.abs(t % TR) < 0.01:
                    nn += 1
                    xs[nn, :] = z[:, 0]

            # Bandpass filter the simulated signal
            signal_filt22 = np.zeros((N, nn))
            for seed in range(N):
                simulated_signal = xs[:, seed]
                simulated_signal = simulated_signal - np.mean(simulated_signal)
                signal_filt22[seed, :] = filtfilt(bfilt, afilt, simulated_signal)

            signal_filt = signal_filt22[:, 20:-20]

            for i in range(N):
                for j in range(N):
                    if nmask[i, j]:
                        GCsim2[sub, i, j] = gcmi_nd_cc(signal_filt22[i, :], signal_filt22[j, :])
                    else:
                        GCsim2[sub, i, j] = 0

        fcsimul = np.mean(GCsim2, axis=0)
        fcsimul[Isubdiag] = 0

        fcsimuli[iteration, :, :] = fcsimul  # Diagonal values to zero
        fittFC[iteration] = np.sqrt(np.nanmean((FCemp[Isubdiag] - fcsimul[Isubdiag]) ** 2))  # Fitting

        corr2FCSC[iteration] = pearsonr(fcsimul[Isubdiag], C[Isubdiag])[0]
        fc = FCemp[Isubdiag]
        idx = np.isnan(fc)
        fc = fc[~idx]
        fcsimdia = fcsimul[Isubdiag]
        fcsimdia = fcsimdia[~idx]
        corr2FCemp_sim[iteration] = pearsonr(fcsimdia, fc)[0]

        # Effective connectivity
        for i in range(N):
            for j in range(N):
                if nmask[i, j]:
                    if C[i, j] > 0 or j == N//2 + i:
                        Cnew[i, j] = Cnew[i, j] + 0.005 * (FCemp[i, j] - fcsimul[i, j])
                        if Cnew[i, j] < 0 or np.isnan(Cnew[i, j]):
                            Cnew[i, j] = 0

        Cnew = Cnew / np.max(Cnew) * 0.2  # Scale effective connectivity to a maximum value of 0.2
        iEffectiveConnectivity[iteration, :, :] = Cnew
