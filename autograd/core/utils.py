import numpy as np

def broadcast_axis(left, right):
    ldim = len(left)
    rdim = len(right)
    maxdim = max(ldim, rdim)

    lshape_new = (1, ) * (maxdim - ldim) + left
    rshape_new = (1, ) * (maxdim - rdim) + right

    assert len(lshape_new) == len(rshape_new)

    left_axes, right_axes = [], []

    for i in range(len(lshape_new)):
        if lshape_new[i] > rshape_new[i]:
            right_axes.append(i)
        elif rshape_new[i] > lshape_new[i]:
            left_axes.append(i)

    return tuple(left_axes), tuple(right_axes)

 
