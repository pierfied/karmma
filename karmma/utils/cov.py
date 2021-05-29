import numpy as np
import healpy as hp
from tqdm.auto import tqdm


def cl2xi_theta(cl, theta):
    """Angular correlation function at separation theta from power spectrum.

    Computes the covariance between pixels at separation of theta from provided power spectrum.
    See https://arxiv.org/pdf/1602.08503.pdf equation 20.

    :param cl: Power spectrum.
    :type cl: array-like (float)
    :param theta: Separation angle in radians.
    :type theta: float or array-like (float)
    :return: xi(theta) - Angular correlation at separation theta.
    :rtype: array-like (float)
    """

    # Convert all array-like input to ndarrays.
    cl = np.asarray(cl)
    theta = np.asarray(theta)

    # Check input sizes.
    if cl.ndim != 1:
        raise Exception('Cl must be a 1D array.')
    if theta.ndim > 1:
        raise Exception('Theta must be a 0D or 1D array.')

    # Get array of l values.
    ells = np.arange(0, len(cl))

    # Compute xi(theta) using Legendre polynomials.
    xi = 1 / (4 * np.pi) * np.polynomial.legendre.legval(np.cos(theta), (2 * ells + 1) * cl)

    return xi


def cl2cov_mat(cl, nside, indices=None, lmax=None, ninterp=10000, log=False, shift=None):
    """Covariance matrix from power spectrum.

    Computes the covariance matrix for the requested pixels from the provided power spectrum.

    :param cl: Power spectrum. Will truncate if len(cl) > lmax + 1.
    :type cl: array-like (float)
    :param nside: Healpix nside parameter.
    :type nside: int
    :param indices: Array of Healpix pixel numbers to compute covariance matrix for. Default: All pixels in the sky.
    :type indices: array-like (int)
    :param lmax: Maximum l mode to include in power spectrum. Default: len(cl) - 1
    :type lmax: float
    :param ninterp: Number of interpolation points for correlation function between 0 and pi. Default: 10,000
    :type ninterp: int
    :param log: Flag to return covariance matrix in log-space. Default: False
    :type log: bool
    :param shift: Shift parameter for log transformation of covariance matrix. Required if log=True. Default: None
    :type shift: float
    :return: Covariance matrix.
    :rtype: array-like (float)
    """

    # If indices is not set default to all pixels.
    if indices is None:
        indices = np.arange(hp.nside2npix(nside))

    # Convert all array-like input to ndarrays.
    cl = np.asarray(cl)
    indices = np.asarray(indices)

    # Check input sizes.
    if cl.ndim != 1:
        raise Exception('Cl must be a 1D array.')
    if indices.ndim != 1:
        raise Exception('Indices must be a 1D array.')
    if len(indices) != len(np.unique(indices)):
        raise Exception('Indices must be unique.')

    # Check that shift parameter is set if log covariance is requested.
    if log is True and shift is None:
        raise Exception('Shift parameter for log covariance transformation must be specified.')

    # Set lmax if not already set.
    if lmax is None:
        lmax = len(cl)

    # Truncate cl if necessary.
    if len(cl) > lmax + 1:
        input_cl = cl[:lmax + 1]
    else:
        input_cl = cl

    # Get the number of pixels.
    npix = len(indices)

    # Get angular coordinates for each pixel.
    theta, phi = hp.pix2ang(nside, indices)
    ang_coord = np.stack([theta, phi])

    # Calculate matrix of separations between pixels.
    ang_sep = np.zeros([npix, npix])
    for i in tqdm(range(npix), desc='Computing angular separations'):
        ang_sep[i, :] = hp.rotator.angdist(ang_coord[:, i], ang_coord)

    # Construct interpolation points for the angular correlation function.
    theta_interp = np.linspace(0, np.pi, ninterp)
    xi_interp = cl2xi_theta(input_cl, theta_interp)

    # Compute covariance matrix using linear interpolation.
    cov = np.interp(ang_sep, theta_interp, xi_interp)

    # Perform log transformation of covariance matrix if requested.
    if log is True:
        cov = np.log(cov / (shift ** 2) + 1)

    return cov
