import numpy as np
import healpy as hp


def add_buffer(mask, buffer):
    """Add a buffer region around a mask.

    :param mask: Mask to be have the buffer added to.
    :type mask: array-like (bool)
    :param buffer: Number of pixels away from mask to include in buffer.
    :type buffer: int
    :return: Mask with added buffer.
    :rtype: array-like (bool)
    """

    # Convert mask to ndarray.
    mask = np.asarray(mask)

    # Check the input types.
    if mask.ndim != 1:
        raise Exception('Mask must be a 1D array.')
    if type(buffer) is not int:
        raise Exception('Buffer argument must be an int type.')

    # Get the nside and npix from the size of the mask.
    npix = len(mask)
    nside = hp.npix2nside(npix)

    # Iteratively compute the pixels to add to the buffer region.
    buffered_pix = np.arange(npix)[mask]
    new_pix = buffered_pix.copy()
    for i in range(buffer):
        # Calculate the new neighbors of the previous layer in the buffer region.
        new_pix = hp.get_all_neighbours(nside, new_pix)
        new_pix = new_pix[new_pix >= 0]
        new_pix = np.setdiff1d(new_pix, buffered_pix)

        # Add the new neighbors to the set of pixels in the buffered mask.
        buffered_pix = np.concatenate([buffered_pix, new_pix])

    # Create the buffered mask.
    buffered_mask = np.zeros_like(mask, dtype=bool)
    buffered_mask[buffered_pix] = True

    return buffered_mask
