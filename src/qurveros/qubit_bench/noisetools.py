"""
This module provides some helper functions to generate time-dependent noise.
"""

import numpy as np
import scipy.special
import scipy.fft
from qurveros.settings import settings


def calculate_alpha_filter_coeffs(*, alpha, n_terms, filter_type):
    """
    Calculates the filter coefficients for the generation of power law noise.

    The method is described in:

    Discrete Simulation of Colored Noise and Stochastic Processes and
    1/f^alpha Power Law Noise Generation by N. Jeremy Kasdin.

    Args:
        alpha (float): The order of the desired power law noise PSD.
        n_terms (int): The number of filter coefficients.
        filter_type (str): The type of the filter. Supported types: FIR, IIR.

    Returns:
        bk (array): An array that contains the coefficients of the numerator
        of the transfer function H(z).
        ak (array): An array that contains the coefficients of the denominator
        of the transfer function H(z).

        See scipy.signal.lfilter for the transfer function specification.
    """

    bk = np.zeros(n_terms)
    ak = np.zeros(n_terms)

    bk[0] = 1
    ak[0] = 1

    if filter_type == "FIR":
        for k in range(1, n_terms):
            c_k = (k - 1 + alpha / 2) / k
            bk[k] = c_k * bk[k - 1]

    else:
        for k in range(1, n_terms):
            c_k = (k - 1 - alpha / 2) / k
            ak[k] = c_k * ak[k - 1]

    return bk, ak


def calculate_baseline_correction(bk):
    """
    Calculates a correction scale that accounts for the distortions introduced
    when the FIR filter is truncated.

    The approximation of the desired power law behavior with a finite
    number of coefficients may introduce errors in the expected magnitude
    of the noise PSD.

    We introduce a baseline correction coefficient that matches the output
    variances when the input is pre-whitened, using first-order
    finite differences. The transfer function of
    the system is H = H_f(z^(-1))(1-z^(-1)). We can find the output variance,
    by taking the PSD = |H|^2 and integrating in the frequency domain.

    For some important properties used in the calculations,
    see: https://dlmf.nist.gov/5.12
    """

    alpha = 2 * bk[1]

    bk_roll = np.roll(bk, -1)
    bk_prod = bk * bk_roll
    actual_variance = 2 * (np.sum(bk**2) - np.sum(bk_prod[:-1]))

    psd_int = 0.5 * scipy.special.beta(0.5 * (3 - alpha), 0.5)
    theoretical_variance = (2 ** (3 - alpha)) * psd_int / np.pi

    baseline_correction = np.sqrt(theoretical_variance / actual_variance)

    return baseline_correction


def get_white_noise_array(*, num_realizations, n_points, rng):
    """
    Generates Gaussian-distributed white noise.
    """

    return rng.standard_normal((num_realizations, n_points))


def get_colored_noise_array(*, num_realizations, n_points, alpha, rng, n_terms=None):
    """
    Generates a colored noise array with prescribed PSD 1/f^alpha based on:

    Discrete Simulation of Colored Noise and Stochastic Processes and
    1/f^alpha Power Law Noise Generation by N. Jeremy Kasdin.

    Args:
        num_realizations (int): The number of noise realizations.
        n_points (int): The total number of samples in the time axis.
        alpha (float): The noise PSD power law exponent.
        rng (np.random): The random number generator.
        n_terms (int): The number of terms for the FIR filter.

    Returns:
        A num_realizations x n_points array containing the noise realizations.

    Note:

    For the consistent simulation of time-dependent noise, the output
    noise must be Wide-Sense Stationary.

    For alpha >=1, the resulting noise is a non-stationary process, where the
    moments grow in time.

    When an IIR filter is used, it was found that such a behavior
    is unavoidable.

    When an FIR filter is used, the moments appear relatively stationary,
    if we skip a couple of initial noise samples. Motivated to remove the
    transient effects from the filtering process, we skip n_terms number of
    samples.

    The baseline correction accounts for scale errors when the filter
    is truncated.
    """

    if n_terms is None:
        n_terms = int(settings.options["FIR_TERMS_FRACTION"] * n_points)

    noise_mat = np.zeros((num_realizations, n_points))
    bk, ak = calculate_alpha_filter_coeffs(
        alpha=alpha, n_terms=n_terms, filter_type="FIR"
    )

    white_noise_mat = get_white_noise_array(
        num_realizations=num_realizations, n_points=n_points + n_terms, rng=rng
    )

    for i in range(num_realizations):
        x_in = white_noise_mat[i, :]
        noise_mat[i, :] = scipy.signal.lfilter(bk, ak, x_in)[n_terms:]

    baseline_correction = calculate_baseline_correction(bk)

    noise_mat = noise_mat * baseline_correction

    return noise_mat


def estimate_psd(sig_array, scale="log", prewhiten=False):
    r"""
    Estimates the PSD using the averaged periodogram method.

    Args:
        sig_array (array): An array of dimensions
        number of realizations x number of time samples
        scale (str): The scale of the angular frequencies and the PSD
        prewhiten (bool): Indicates to apply a first-order finite difference
        on the data, so that the resulting PSD is multiplied by $\omega^2$.

    Returns:
        A tuple containing the angular frequencies and the estimated PSD.

    Note:

    The reader is referred to the excellent book
    Modern Spectral Estimation: Theory and Application by Steven M. Kay.
    The method is described in section 4.4.

    Care must be taken on the interpretation of the results. If the estimated
    PSD contains appreciable weight in the low frequencies, the windowing of
    the sequence might result to misidentification of the function scale.

    For more details, see:

    Discrete Simulation of Colored Noise and Stochastic Processes and
    1/f^alpha Power Law Noise Generation by N. Jeremy Kasdin.
    """

    if prewhiten:
        sig_array = np.diff(sig_array, axis=1)

    n_points = sig_array.shape[1]

    sig_array_fft = scipy.fft.rfft(sig_array, axis=1)

    psd_est = 1 / n_points * np.abs(sig_array_fft) ** 2
    psd_est = np.mean(psd_est, axis=0)

    ang_freqs = (2 * np.pi / n_points) * np.arange(n_points)

    ang_freqs = ang_freqs[: int(n_points / 2) + 1]

    if scale == "log":
        psd_est = np.log10(psd_est[1:])
        ang_freqs = np.log10(ang_freqs[1:])

    return ang_freqs, psd_est
