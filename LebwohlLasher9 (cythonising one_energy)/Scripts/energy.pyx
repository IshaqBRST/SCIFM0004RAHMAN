# Import the required C library for math functions
from libc.math cimport cos

# Define the function with types for its arguments and internal variables
cpdef double one_energy(double[:, :] arr, int ix, int iy, int nmax):
    cdef:
        int ixp, ixm, iyp, iym
        double en = 0.0
        double ang

    # Compute neighbor indices with periodic boundary conditions
    ixp = (ix + 1) % nmax
    ixm = (ix - 1) % nmax
    iyp = (iy + 1) % nmax
    iym = (iy - 1) % nmax

    # Calculate energy contributions from neighboring cells
    ang = arr[ix, iy] - arr[ixp, iy]
    en += 0.5 * (1.0 - 3.0 * cos(ang) ** 2)
    ang = arr[ix, iy] - arr[ixm, iy]
    en += 0.5 * (1.0 - 3.0 * cos(ang) ** 2)
    ang = arr[ix, iy] - arr[ix, iyp]
    en += 0.5 * (1.0 - 3.0 * cos(ang) ** 2)
    ang = arr[ix, iy] - arr[ix, iym]
    en += 0.5 * (1.0 - 3.0 * cos(ang) ** 2)

    return en
