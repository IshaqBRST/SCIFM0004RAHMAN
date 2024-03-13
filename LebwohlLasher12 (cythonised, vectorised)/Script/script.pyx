# energy_functions.pyx
cimport cython
from libc.math cimport cos
import numpy as np
cimport numpy as cnp
from libc.math cimport cos
import numpy as np
cimport numpy as cnp
cimport cython
from libc.math cimport cos

cnp.import_array()  # Necessary for NumPy operations in Cython

cpdef initdat(int nmax):

    cdef cnp.ndarray[cnp.float64_t, ndim=2] arr
    arr = np.random.random_sample((nmax, nmax)) * 2.0 * np.pi
    return arr

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
        int i, j
        cnp.ndarray[cnp.float64_t, ndim=2] right, left, up, down
        cnp.ndarray[cnp.float64_t, ndim=2] en_right, en_left, en_up, en_down
        double total_energy
    
    # Convert memory views to numpy arrays for the operations
    arr_np = np.asarray(arr)

    # Compute shifted arrays for periodic boundary conditions
    right = np.roll(arr_np, -1, axis=1)
    left = np.roll(arr_np, 1, axis=1)
    up = np.roll(arr_np, -1, axis=0)
    down = np.roll(arr_np, 1, axis=0)

    # Calculate the energy contributions
    en_right = 0.5 * (1.0 - 3.0 * np.cos(arr_np - right) ** 2)
    en_left = 0.5 * (1.0 - 3.0 * np.cos(arr_np - left) ** 2)
    en_up = 0.5 * (1.0 - 3.0 * np.cos(arr_np - up) ** 2)
    en_down = 0.5 * (1.0 - 3.0 * np.cos(arr_np - down) ** 2)

    # Sum all contributions
    total_energy = np.sum(en_right + en_left + en_up + en_down)
    
    return total_energy


# cython: language_level=3
import numpy as np
cimport numpy as cnp
from libc.math cimport cos, sin
from scipy.linalg.cython_lapack cimport dsyevd

# Necessary Cython and numpy imports
cimport cython
import numpy as np
cimport numpy as cnp
from libc.math cimport cos, sin

# Ensuring numpy arrays can be used
# Assuming you have included necessary Cython and NumPy cimport statements
import numpy as np
cimport cython
cimport numpy as cnp
from libc.math cimport cos, sin

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double get_order(double[:, :] arr, int nmax):
    cdef:
        int i, j
        double[:, :] Qab = np.zeros((3, 3), dtype=np.float64)
        double[:, :] Qxx = np.empty((nmax, nmax), dtype=np.float64)
        double[:, :] Qyy = np.empty((nmax, nmax), dtype=np.float64)
        double[:, :] Qxy = np.empty((nmax, nmax), dtype=np.float64)
        double max_eigenvalue

    # Compute components of the Q tensor
    for i in range(nmax):
        for j in range(nmax):
            Qxx[i, j] = (3.0 * cos(arr[i, j]) ** 2 - 1) / 2.0
            Qyy[i, j] = (3.0 * sin(arr[i, j]) ** 2 - 1) / 2.0
            Qxy[i, j] = 1.5 * cos(arr[i, j]) * sin(arr[i, j])

    # Averaging over the lattice
    Qab[0, 0] = np.mean(Qxx)
    Qab[1, 1] = np.mean(Qyy)
    Qab[2, 2] = -0.5 * (Qab[0, 0] + Qab[1, 1])
    Qab[0, 1] = Qab[1, 0] = np.mean(Qxy)
    Qab[0, 2] = Qab[2, 0] = 0.0  # No xz or yz components in 2D problem
    Qab[1, 2] = Qab[2, 1] = 0.0

    # Use NumPy for eigenvalues calculation
    eigenvalues = np.linalg.eigvalsh(Qab)
    max_eigenvalue = np.max(eigenvalues)

    return max_eigenvalue

# cython: language_level=3
import numpy as np
cimport numpy as cnp
cimport cython
from libc.stdlib cimport rand, RAND_MAX, srand
from libc.time cimport time
from libc.math cimport exp

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double MC_step(double[:, :] arr, double Ts, int nmax):
    # Seed random number generator
    srand(time(NULL))
    
    cdef:
        int accept = 0
        int i, j, ix, iy
        double en0, en1, deltaE, boltz, ang, randVal
        double scale = 0.1 + Ts
    
    for i in range(nmax):
        for j in range(nmax):
            ix = rand() % nmax
            iy = rand() % nmax
            
            # Assuming a normal distribution with mean 0 and std dev `scale`
            # NumPy's random.normal is used here as C doesn't have a direct equivalent
            ang = np.random.normal(0, scale)
            
            en0 = one_energy(arr, ix, iy, nmax)
            arr[ix, iy] += ang
            en1 = one_energy(arr, ix, iy, nmax)
            
            if en1 <= en0:
                accept += 1
            else:
                deltaE = en1 - en0
                boltz = exp(-deltaE / Ts)
                
                # Generate a random float between 0 and 1
                randVal = rand() / float(RAND_MAX)
                
                if boltz >= randVal:
                    accept += 1
                else:
                    arr[ix, iy] -= ang
    
    return accept / (nmax * nmax)




