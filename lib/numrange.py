import jax.numpy as np
from jax import grad, vmap, jit, pmap
import sys

import numpy as onp
from numpy.random import rand, randn, seed
import matplotlib.pyplot as plt
from numpy.linalg import inv
import numpy.linalg as la
import matplotlib
np.set_printoptions(precision=2)
matplotlib.rc('text.latex', preamble=r'\usepackage{amsmath},\usepackage{amssymb}')

""" Numerical range of matrix M = [[A,B],[C,D]] """

def numerical_range(A, N=int(1e5)):
    xs = sphere(N, A.shape[0])
    W = [(A @ x) @ np.conj(x) for x in xs]
    return W

def block_to_2x2(J, x, y):
    A,B,C,D = J
    quad = lambda x,A,y: np.conj(x) @ A @ y
    M = np.asarray([[quad(x, A, x), quad(x, B, y)], \
                   [quad(y, C, x), quad(y, D, y)]])
    return M

@jit
def qnum(J,x,y):
    M = block_to_2x2(J)
    return eig2x2(M)
qnum = jit(vmap(qnum, (None, 0, 0)))

def eig2x2(A):
    (a,b),(c,d) = A
    root = lambda tr, det: (tr/2 + np.sqrt(tr**2 - 4*det)/2, 
                            tr/2 - np.sqrt(tr**2 - 4*det)/2 )
    return root(a+d, a*d-b*c)

def sphere(num_samples, n):
    x = onp.random.randn(num_samples, n) \
        + onp.random.randn(num_samples, n)*1j
    x /= la.norm(x, axis=1)[:,np.newaxis]
    return x

def quadratic_numerical_range(A, B, C, D, N=int(1e6)):

    xs = sphere(N, A.shape[0])
    ys = sphere(N, D.shape[0])
    return np.hstack(qnum((A,B,C,D), xs, ys))
  
def eig_minmax(A):
    idx = [0,-1]
    return np.sort(la.eigvals(A))[idx]

def plt_hline(ax,h, mirror=True, **pltargs):
    ax.plot([-10,10],[h,h], '--k', **pltargs)
    if mirror:
        ax.plot([-10,10],[-h,-h], '--k', **pltargs)

def plt_vline(ax, v, **pltargs):
    ax.plot([v, v], [-10, 10], '--k', **pltargs)
    
def rand_4x4(seed=214):
    onp.random.seed(seed)
    ri = lambda x: onp.random.randint(-10*x, 10*x)/10
    a11 = np.abs(ri(10))
    a22 = np.abs(ri(10))
    d11 = np.abs(ri(10))
    d22 = np.abs(ri(10))

    a12, d12 = ri(1), ri(1)

    b11 = ri(1) 
    b12 = ri(1)
    b21 = ri(1)
    b22 = ri(1)

    c11 = ri(1) 
    c12 = ri(1)
    c21 = ri(1)
    c22 = ri(1)
  
    A = np.array([[a11, a12],
                [a12, a22]])
    D = np.array([[d11, d12],
                [d12, d22]])
    B = np.array([[b11, b12],
                [b21, b22]])
    C = np.array([[c11, c12],
                [c21, c22]])

    J = (A,B,C,D)
    return J

def print_info(J):
    A,B,C,D = J
    print("J     :", block(*J))
    print("J eigs:", onp.linalg.eigvals(block(*J)))
    print("tr(J)", onp.trace(block(*J)))
    print("det(J)", onp.linalg.det(block(*J)))
    print("A eigs:", onp.linalg.eigvals(A))
    print("D eigs:", onp.linalg.eigvals(D))
    print("|A|   :", onp.linalg.norm(A))
    print("|B|   :", onp.linalg.norm(B))
    print("|C|   :", onp.linalg.norm(C))
    print("|D|   :", onp.linalg.norm(D))

def scan(start=214, end=214):
    """ Scans for 'nice looking' spectra for 4x4 general sum game """
  
    for i in range(start, end+1):
        sys.stdout.write('.')
        J = rand_4x4(i)
        A,B,C,D = J
        if np.all(np.real(onp.linalg.eigvals(block(*J))) > 0.5): continue
        if np.all(np.imag(onp.linalg.eigvals(block(*J))) < 0.1): continue

        print()
        print("seed  :", i)
        print_info(J)
    
        W = quadratic_numerical_range(*J, )
        fig, ax = plt.subplots()
        plt_W(ax, W)
        plt_eigs(ax, block(*J))
        plt_eigs(ax, A)
        plt_eigs(ax, D)
        plt.xlim([0,10])
        plt.ylim([-1,1])
        plt.show()

""" Tools """

def PD(n, eps=1e-3):
    A = onp.random.randn(n,n)
    A = A.T @ A
    return A/la.norm(A) + eps*np.eye(n)

def eigs(A):
    return la.eigvals(A)

def pop(args, var, default=0):
    try: 
        return args.pop(var)
    except: 
        return default

def block(A,B,C,D):
    return onp.block([[A,B],[C,D]])

def zero_sum_game(seed=3, nx=4, ny=5, scale = .5):
    onp.random.seed(seed)
    A = PD(nx)
    B = randn(nx, ny); 
    B /= la.norm(B); 
    B *= scale
    C = -B.T 
    D = PD(ny)
    return A,B,C,D

def general_sum_game(seed=3, seed_=4, nx=4, ny=5, scale=.1, full=False, sup=0.0):
    onp.random.seed(seed)
    A1 = PD(nx)
    B = randn(nx, ny)
    B /= la.norm(B); 
    C = randn(ny, nx)
    C /= la.norm(C); 
    B *= scale
    C *= scale
    D2 = PD(ny)
    if full:
        onp.random.seed(seed_)
        A2=randn(ny,ny) #PD(ny)+sup*np.eye(ny)
        D1=randn(nx,nx) #PD(nx)+sup*np.eye(nx)
        return A1, B, C, D2, A2, D1
    else:
        return A1,B,C,D2

def plt_setup(ax):
    ax.set_aspect('equal')
    ax.grid()
    ax.legend(bbox_to_anchor=[1.2, 1])
    ax.set_ylim([-1,1])

def plt_eigs(ax, A, **pltargs):
    e = eigs(A)
    shift = pop(pltargs, 'y_shift', 0)
    ax.plot(np.real(e), np.imag(e) + shift, 'o', **pltargs)

def plt_W(ax, W, **pltargs):
    #   ax.hist(W, bins=128, density=True, **pltargs)
    ax.plot(np.real(W), np.imag(W), ',', **pltargs)

def split_block(J, r):
    A = J[0:r,0:r]
    B = J[0:r,r:]
    C = J[r:,0:r]
    D = J[r:,r:]
    return A,B,C,D

def plt_origin(ax, **pltargs):
    ax.plot([-10,10],[0,0], ':k', **pltargs)
    ax.plot([0,0],[-10,10], ':k', **pltargs)
