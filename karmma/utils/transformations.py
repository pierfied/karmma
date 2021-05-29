import numpy as np
import healpy as hp
from tqdm.auto import tqdm


def shear2conv(g1, g2, lmax=None):
    """Convert shear map to convergence map.

    Computes convergence map from shear map components via spherical harmonics.
    See https://arxiv.org/pdf/1708.01535.pdf equation 9.

    :param g1: First shear component map.
    :type g1: array-like (float)
    :param g2: Second shear component map.
    :type g2: array-like (float)
    :param lmax: Maximum l mode to include in spherical harmonic transformation. Default: 3*nside - 1
    :type lmax: int
    :return: Convergence map.
    :rtype: array-like (float)
    """

    # Convert all array-like input to ndarrays.
    g1 = np.asarray(g1)
    g2 = np.asarray(g2)

    # Check input sizes.
    if len(g1) != len(g2):
        raise Exception('First and second shear components are not the same length.')
    if g1.ndim != 1 or g2.ndim != 1:
        raise Exception('Input shear map components must be 1D arrays.')

    # Get the nside of the map.
    nside = hp.get_nside(g1)

    # Set the T map to zero.
    gt = np.zeros_like(g1)

    # Get E mode shear map alms using spin-2 spherical harmonics.
    _, gelm, _ = hp.map2alm([gt, g1, g2], lmax=lmax)

    # Get the lmax and l, m values of the alms.
    lmax = hp.Alm.getlmax(len(gelm))
    l, m = hp.Alm.getlm(lmax)

    # Construct the E mode convergence map alms and set to zero for l = 0,1 to avoid divide by zero.
    good_ls = l > 1
    l = l[good_ls]
    kelm = np.zeros_like(gelm)
    kelm[good_ls] = - np.sqrt(l * (l + 1) / ((l + 2) * (l - 1))) * gelm[good_ls]

    # Get real-space convergence map using spherical harmonics.
    k = hp.alm2map(kelm, nside, lmax=lmax, verbose=False)

    return k


def conv2shear(k, lmax=None):
    """Convert convergence map to shear map.

    Computes shear map components from convergence map via spherical harmonics.
    See https://arxiv.org/pdf/1708.01535.pdf equation 9.

    :param k: Convergence map.
    :type k: array-like (float)
    :param lmax: Maximum l mode to include in spherical harmonic transformations. Default: 3*nside - 1
    :type lmax: int
    :return: Shear map components.
    :rtype: Tuple of array-like (float)
    """

    # Convert all array-like input to ndarrays.
    k = np.asarray(k)

    # Check input size.
    if k.ndim != 1:
        raise Exception('Input convergence map must be a 1D array.')

    # Get the nside of the map.
    nside = hp.get_nside(k)

    # Convert convergence map to alms via spherical harmonics.
    kelm = hp.map2alm(k, lmax=lmax)

    # Get the lmax and l,m values of the alms.
    lmax = hp.Alm.getlmax(len(kelm))
    l, m = hp.Alm.getlm(lmax)

    # Create the zero-initialized shear alms.
    gtlm = np.zeros_like(kelm)
    gelm = np.zeros_like(kelm)
    gblm = np.zeros_like(kelm)

    # Compute the E mode shear map alms and set to zero for l = 0 to avoid divide by zero.
    good_ls = l > 0
    l = l[good_ls]
    gelm[good_ls] = - np.sqrt((l + 2) * (l - 1) / (l * (l + 1))) * kelm[good_ls]

    # Get the real-space shear map components using spin-2 spherical harmonics.
    _, g1, g2 = hp.alm2map([gtlm, gelm, gblm], nside, lmax=lmax, verbose=False)

    return g1, g2


def conv2shear_mats(nside, mask=None, lmax=None):
    """Create matrices to perform convergence to shear transformations.

    Builds the matrices by individually computing each convergence pixel's contribution to all shear pixels.

    :param nside: Healpix nside parameter.
    :type nside: int
    :param mask: Healpix pixel mask to build matrices for. Default: All pixels in the sky.
    :type indices: array-like (bool)
    :param lmax: Maximum l mode to include in transformation. Default: 3*nside - 1
    :type lmax: int
    :return: Transformation matrices for convergence to shear map components.
    :rtype: Tuple of array-like (float)
    """

    # Get the number of pixels for the full-sky map.
    npix = hp.nside2npix(nside)

    # If mask is not set assume full-sky.
    indices = np.arange(hp.nside2npix(nside))
    if mask is not None:
        indices = indices[mask]

    # Convert all array-like input to ndarrays.
    indices = np.asarray(indices)

    # Check input.
    if indices.ndim != 1:
        raise Exception('Indices must be a 1D array.')
    if len(indices) != len(np.unique(indices)):
        raise Exception('Indices are not unique.')

    # Get the number of pixels of interest.
    ninds = len(indices)

    # Create the transformation matrices.
    A1 = np.zeros((ninds, ninds))
    A2 = np.zeros_like(A1)

    # Build transformation matrices by calculating each convergence pixel's contribution to each shear pixel.
    for i, ind in enumerate(tqdm(indices, desc='Computing k2g matrix')):
        # Create a convergence map with only the current pixel set to 1.
        k = np.zeros(npix)
        k[ind] = 1

        # Calculate shears.
        g1, g2 = conv2shear(k, lmax)

        # Set matrix value for this convergence pixels contribution to all shear pixels.
        A1[:, i] = g1[indices]
        A2[:, i] = g2[indices]

    return A1, A2
