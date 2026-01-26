import numpy as np

def rle_decode(mask_rle, shape):
    """
        mask_rle: string like '31632 6 31886 10 ...'
        shape: (height, width) of the image
        returns: 2D numpy array mask (0=background, 1=cell)
        """
    if mask_rle == '' or mask_rle is np.nan:
        return np.zeros(shape,dtype=np.uint8)

    s = mask_rle.split()
    starts, length = [np.array(x,dtype=int) for x in (s[0::2],s[1::2])]
    starts -= 1
    ends = starts + length

    image = np.zeros(shape[0]*shape[1],dtype=np.uint8)
    for lo, hi in zip(starts,ends):
        image[lo:hi]=1
    return image.reshape(shape,order='F') #Column major