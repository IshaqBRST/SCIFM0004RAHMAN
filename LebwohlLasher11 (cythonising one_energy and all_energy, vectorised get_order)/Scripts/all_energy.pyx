# energy_functions.pyx
cimport cython
from libc.math cimport cos

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double one_energy(double[:, :] arr, int ix, int iy, int nmax):
    cdef:
        double en = 0.0
        int ixp, ixm, iyp, iym
        double ang
    
    ixp = (ix + 1) % nmax
    ixm = (ix - 1) % nmax
    iyp = (iy + 1) % nmax
    iym = (iy - 1) % nmax
    
    # Calculate energies considering periodic boundary conditions
    ang = arr[ix, iy] - arr[ixp, iy]
    en += 0.5 * (1.0 - 3.0 * cos(ang) ** 2)
    
    ang = arr[ix, iy] - arr[ixm, iy]
    en += 0.5 * (1.0 - 3.0 * cos(ang) ** 2)
    
    ang = arr[ix, iy] - arr[ix, iyp]
    en += 0.5 * (1.0 - 3.0 * cos(ang) ** 2)
    
    ang = arr[ix, iy] - arr[ix, iym]
    en += 0.5 * (1.0 - 3.0 * cos(ang) ** 2)
    
    return en

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double all_energy(double[:, :] arr, int nmax):
    cdef:
        double enall = 0.0
        int i, j
    
    for i in range(nmax):
        for j in range(nmax):
            enall += one_energy(arr, i, j, nmax)
    
    return enall
