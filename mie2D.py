import numpy as np
import scipy.special as sp

def round_to_int(m):
    
    # m = 1.01 --> int(m) = 1 --> m - int(m) = 0.01
    # m = 0.99 --> int(m) = 0 --> m - int(m) = 0.99
    
    if (m-int(m) >= 0.5):
        mr = np.ceil(m).astype(int)
    else:
        mr = np.floor(m).astype(int)
    
    return mr

def mie_solution(X, Y, xc, yc, morder, R, k0, epsrmat, theta_rot):
    
    ny, nx = X.shape
    
    k1 = k0*np.sqrt(epsrmat)
    
    theta_rot = -theta_rot # rotation in xy-coordinate with theta_rot
    tt = np.deg2rad(theta_rot)
    
    Xt, Yt = X - xc, Y - yc
    Xrot = Xt*np.cos(tt) - Yt*np.sin(tt)
    Yrot = Xt*np.sin(tt) + Yt*np.cos(tt)
    Xfinal, Yfinal = np.copy(Xrot), np.copy(Yrot)
    
    Dist2D = np.sqrt(Xfinal**2 + Yfinal**2)
    Dist1D = Dist2D.flatten()
    
    Phi2D = np.arctan2(Yfinal, Xfinal)
    solution2D = np.zeros([ny, nx, 2*morder+1], dtype=np.complex128)

    for m in range(-morder, morder+1):
        
        # step 1: calculate the Fourier coefficients (FC)
        Eh1D = calculate_FC(Dist1D, R, k0, k1, m)
        Eh2D = Eh1D.reshape([ny, nx])
        
        # step 2: calculate the phase
        Phase2D = np.exp(1j*m*Phi2D)
        
        # step 3: summation
        solution2D[:, :, m] = Eh2D*Phase2D
    
    solution2D = np.sum(solution2D, axis=2)
    
    return solution2D

def calculate_outside(r, order, k0, k1, Am, Bm):
    
    Eh_out = Am*sp.jv(order, k0*r) + Bm*sp.hankel1(order, k0*r)
    
    return Eh_out

def calculate_inside(r, order, k1, Cm):
    
    Eh_in = Cm*sp.jv(order, k1*r)
    
    return Eh_in

def calculate_FC(dist1D, R, k0, k1, order):
     
    k0R, k1R = k0*R, k1*R
    RI = k1/k0
    
    Am = 1j**order
    
    b1 = sp.jv(order, k1R)*sp.jv(order+1, k0R) - RI*sp.jv(order+1, k1R)*sp.jv(order, k0R)
    b2 = sp.jv(order, k1R)*sp.hankel1(order+1, k0R) - RI*sp.jv(order+1, k1R)*sp.hankel1(order, k0R)
    Bm = -(b1/b2)*Am
    
    c1 = b2
    Cm = (-2j/(np.pi*k0R))*(1/c1)*Am
    
    Eh_1D = np.zeros_like(dist1D, dtype=np.complex128)
    
    # outside an object
    Eh_1D[dist1D>R] = calculate_outside(dist1D[dist1D>R], order, k0, k1, Am, Bm)
    
    # inside an object
    Eh_1D[dist1D<=R] = calculate_inside(dist1D[dist1D<=R], order, k1, Cm)

    return Eh_1D

if __name__ == '__main__':
    pass