import numpy as np
import healpy as hp
from tqdm.auto import tqdm
from scipy.special import eval_legendre


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


def cl2cov_mat(cl, nside, mask=None, lmax=None, ninterp=10000, log=False, shift=None):
    """Covariance matrix from power spectrum.

    Computes the covariance matrix for the requested pixels from the provided power spectrum.

    :param cl: Power spectrum. Will truncate if len(cl) > lmax + 1.
    :type cl: array-like (float)
    :param nside: Healpix nside parameter.
    :type nside: int
    :param mask: Healpix mask to compute the covariance matrix over. Default: All pixels in the sky.
    :type mask: array-like (bool)
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

    # If mask is not set default to all pixels.
    indices = np.arange(hp.nside2npix(nside))
    if mask is not None:
        indices = indices[mask]

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


def lognorm_to_gauss_cl(lognorm_cl, shift, lmax=None):
    # Discard l > lmax if lmax is provided.
    if lmax is None:
        lmax = len(lognorm_cl) - 1
    else:
        lognorm_cl = lognorm_cl[:lmax + 1]

    # Get the Gauss-Legendre quadrature sample points and weights.
    mu, w = np.polynomial.legendre.leggauss(2 * lmax)

    # Compute the integrand.
    ell = np.arange(lmax + 1)
    integrand = np.log(np.polynomial.legendre.legval(mu, (2 * ell + 1) * lognorm_cl / (4 * np.pi * (shift ** 2))) + 1)

    # Compute the integral.
    gauss_cl = 2 * np.pi * np.sum(integrand[None, :] * eval_legendre(ell[:, None], mu[None, :]) * w[None, :], axis=-1)

    # Clip negative values.
    gauss_cl = np.maximum(0, gauss_cl)

    return gauss_cl
