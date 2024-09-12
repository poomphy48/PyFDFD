import numpy as np
import scipy.special as spfn

def round_to_int(m):
    
    if (m-int(m) >= 0.5):
        mr = np.ceil(m).astype(int)
    else:
        mr = np.floor(m).astype(int)
    
    return mr

def compute_G(XHH, YHH, domain_rx, k0, gamma0, typeHankel):

    # The order of the Hankel function
    ORDER = 0
    
    ## HH domain
    XHHf, YHHf = XHH.flatten(), YHH.flatten()
   
    ##### old version (It supports for 1D or 2D measurment plane)
    # Xf_rx, Yf_rx, dim_rx, _ = domain_rx
    # nx_rx, ny_rx = dim_rx
    # nrx = nx_rx*ny_rx
    
    ##### this version (It supports only 1D measurment plane and the plane is placed parallel to y-axis.)
    
    ####################################################
    # version 1: before 19 April 2024
#     xc_rx, ymin_rx, ymax_rx, nrx = domain_rx
#     yrx = np.linspace(ymin_rx, ymax_rx, nrx)
    
#     X_rx, Y_rx = np.meshgrid(xc_rx, yrx)
#     Xf_rx, Yf_rx = X_rx.flatten(), Y_rx.flatten()
    
    # version 2: after 19 April 2024
    Xf_rx, Yf_rx, nrx = domain_rx
    ####################################################
    
    XfT_rx, YfT_rx = Xf_rx.reshape([nrx, 1]), Yf_rx.reshape([nrx, 1])

    # calculate the distance between target-positions and receiver-positions
    # broadcasting in numpy array
    Dist2D = np.sqrt((XfT_rx-XHHf)**2 + (YfT_rx-YHHf)**2)

    ###############  Debug  ###################################################
    # import matplotlib.pyplot as plt
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5))
    # fig1 = ax1.imshow(Xf_rx.reshape([ny_rx, nx_rx]), cmap='jet'); ax1.set_title('X_rx\n(matrix)'); plt.colorbar(fig1, ax=ax1)
    # fig2 = ax2.imshow(Yf_rx.reshape([ny_rx, nx_rx]), cmap='jet'); ax2.set_title('Y_rx\n(matrix)'); plt.colorbar(fig2, ax=ax2); plt.show()
    # plt.imshow(Dist2D, cmap='jet'); plt.title('Dist2D\n(matrix)'); plt.colorbar(); plt.show()
    ###########################################################################

    if typeHankel == '1st':
        Green2D = (1j/4)*(gamma0**2)*spfn.hankel1(ORDER, k0*Dist2D)
        print('LS_computation at line 49: +(1j/4) --> it is okay because the LS solution matches to the HH solution with very large computational size.')

    elif typeHankel == '2nd':
        Green2D = (-1j/4)*(gamma0**2)*spfn.hankel2(ORDER, k0*Dist2D)
        
    return Green2D

def compute_mask2D(MatHH2D):
    
    nyHH, nxHH = MatHH2D.shape
    
    # build the mask2D for numerical intergration with the trapezoidal rule
    MaskHH2D = np.ones([nyHH, nxHH])
    
    # corners
    MaskHH2D[0, 0] = 0.25
    MaskHH2D[0, -1] = 0.25
    MaskHH2D[-1, 0] = 0.25
    MaskHH2D[-1, -1] = 0.25
    
    # sides
    MaskHH2D[0, 1:-2] = 0.50
    MaskHH2D[-1, 1:-2] = 0.50
    MaskHH2D[1:-2, 0] = 0.50
    MaskHH2D[1:-2, -1] = 0.50
    
    return MaskHH2D
