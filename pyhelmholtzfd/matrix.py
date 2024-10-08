import numpy as np
from scipy.sparse import csr_matrix

from .stencils import generate_index, laplace_stencil, wavenumber_stencil, oneway_stencil

class OperationMatrix:
    
    def __init__(self, nxHH, nyHH, nly):
        
        self.nxHH, self.nyHH, self.nly = nxHH, nyHH, nly
        self.nx = nxHH + 2*nly
        self.ny = nyHH + 2*nly        
        self.n = self.nx*self.ny
        
    def configuration(self, epsrbg, gamma0, typeOWeqn, hhorder, oworder, typeWeight):
        
        nx = self.nx
        
        if typeOWeqn == 'MUR1':
            
            if oworder == '1st-onesided':
                # 1st-order Mur with 1st-order one-sided FD scheme
                self.FD_side = 'side_MUR1_1stFD'
                self.FD_corner = 'corner_MUR1_1stFD'
                self.Idx_nb_for_side = np.array([[1], [-1], [nx], [-nx]])
                self.Idx_nb_for_corner = np.array([[1, nx], [-1, nx], [1, -nx], [-1, -nx]])
            
            elif oworder == '2nd-onesided':
                # 1st-order Mur with 2nd-order one-sided FD scheme
                self.FD_side = 'side_MUR1_2ndFD'
                self.FD_corner = 'corner_MUR1_2ndFD' 
                self.Idx_nb_for_side = np.array([[1, 2], [-1, -2], [nx, 2*nx], [-nx, -2*nx]])
                self.Idx_nb_for_corner = np.array([[1, 2, nx, 2*nx], [-1, -2, nx, 2*nx], [1, 2, -nx, -2*nx], [-1, -2, -nx, -2*nx]])
        
        elif typeOWeqn == 'MUR2':
            
            if oworder == '1st-onesided':
                # 2nd-order Mur with 1st-order one-sided FD scheme
                self.FD_side = 'side_MUR2_1stFD'
                self.FD_corner = 'corner_MUR1_1stFD'
                self.Idx_nb_for_side = np.array([[1, -nx, nx], [-1, -nx, nx], [nx, -1, 1], [-nx, -1, 1]])
                self.Idx_nb_for_corner = np.array([[1, nx], [-1, nx], [1, -nx], [-1, -nx]])
            
            elif oworder == '2nd-onesided':
                # 2nd-order Mur with 2nd-order one-sided FD scheme
                self.FD_side = 'side_MUR2_2ndFD'
                self.FD_corner = 'corner_MUR1_2ndFD'
                self.Idx_nb_for_side = np.array([[1, 2,-nx, nx], [-1, -2, -nx, nx], [nx, 2*nx, -1, 1], [-nx, -2*nx, -1, 1]])
                self.Idx_nb_for_corner = np.array([[1, 2, nx, 2*nx], [-1, -2, nx, 2*nx], [1, 2, -nx, -2*nx], [-1, -2, -nx, -2*nx]])
        
        self.epsrbg, self.gamma0 = epsrbg, gamma0
        self.typeOWeqn = typeOWeqn
        self.hhorder = hhorder
        self.oworder = oworder
        self.typeWeight = typeWeight
        
        if typeOWeqn != 'MUR1' and typeOWeqn != 'MUR2':            
            raise Exception('typeOWeqn for the one-way wave equation are:\n1. MUR1\n2. MUR2')
        
        if hhorder != '2nd-central' and hhorder != '4th-central' and hhorder != '9point':            
            raise Exception('The FD scheme for the Helmholtz equation (hhorder) are:\n1. 2nd-central\n2. 4th-central\n3. 9 points')
        
        if oworder != '1st-onesided' and oworder != '2nd-onesided':            
            raise Exception('The FD scheme for the one-way wave equation (oworder) are:\n1. 1st-onesided\n2. 2nd-onesided')

        if typeWeight != 'linear' and typeWeight != 'quadratic' and typeWeight != 'cube' and typeWeight != 'zero' and typeWeight != 'one':
            raise Exception('typeWeight are:\n1. linear\n2. quadratic\n3. cubic\n4. zero\n5. one')
            
    def medium(self, Mat2D):
        
        nx, ny, nly = self.nx, self.ny, self.nly
        
        if Mat2D.shape == (ny, nx):            
            self.Mat1D_HH = (Mat2D[nly:-nly, nly:-nly]).flatten()
            
        else:
            raise Exception('The size of Mat2D is not match to (ny, nx).')
        
    def build_A1(self):
        
        '''
        The operation matrix A for inner zone        
        '''
        
        gamma0 = self.gamma0
        nx, ny, nly = self.nx, self.ny, self.nly
        n = self.n
        
        # part 1/2: spatial-operation matrix, (d^2/dx^2 + d^2/dy^2)
        if self.hhorder == '2nd-central':     
        
            # coefficients = -4, 1, 1, 1, 1
            idx_pos = (((np.arange(n)).reshape([ny, nx]))[nly:-nly, nly:-nly]).flatten()
            ir, ic, v = laplace_stencil('2nd-central', idx_pos, nx)
            A1_spatial = csr_matrix((v, (ir, ic)), shape=(n, n), dtype=np.complex64)
        
        elif self.hhorder == '4th-central' and nly != 1:
            
            # coefficients = -5, 16/12, 16/12, 16/12, 16/12, -1/12, -1/12, -1/12, -1/12
            idx_pos = (((np.arange(n)).reshape([ny, nx]))[nly:-nly, nly:-nly]).flatten()
            ir, ic, v = laplace_stencil('4th-central', idx_pos, nx)
            A1_spatial = csr_matrix((v, (ir, ic)), shape=(n, n), dtype=np.complex64)
            
        elif self.hhorder == '4th-central' and nly == 1:      
        
            # phy = inside phy. domain + boundary of phy. domain
            irow_phy, icol_phy, val_phy = [], [], []
            
            # inside phy. domain: coefficients = -5, 16/12, 16/12, 16/12, 16/12, -1/12, -1/12, -1/12, -1/12
            idx_pos = (((np.arange(n)).reshape([ny, nx]))[nly+1:-nly-1, nly+1:-nly-1]).flatten()
            ir, ic, v = laplace_stencil('4th-central', idx_pos, nx)
            irow_phy, icol_phy, val_phy = np.append(irow_phy, ir), np.append(icol_phy, ic), np.append(val_phy, v)
            
            # boundary of phy. domainn: coefficients = -4, 1, 1, 1, 1
            ix1, ix2 = nly, (nx - 1) - nly
            iy1, iy2 = nly, (ny - 1) - nly
            
            idx_corner, idx_l, idx_r, idx_b, idx_t = generate_index(ix1, ix2, iy1, iy2, nx)
            idx_pos = np.concatenate((idx_corner, idx_l, idx_r, idx_b, idx_t))
            ir, ic, v = laplace_stencil('2nd-central', idx_pos, nx)
            irow_phy, icol_phy, val_phy = np.append(irow_phy, ir), np.append(icol_phy, ic), np.append(val_phy, v)
            
            A1_spatial = csr_matrix((val_phy, (irow_phy, icol_phy)), shape=(n, n), dtype=np.complex64)
        
        # part 2/2: wavenumber term, (gamma0**2)*epsr
        idx_pos = (((np.arange(n)).reshape([ny, nx]))[nly:-nly, nly:-nly]).flatten()
        ir, ic, v = wavenumber_stencil(idx_pos, self.Mat1D_HH, gamma0)
        A1_wavenumber = csr_matrix((v, (ir, ic)), shape=(n, n), dtype=np.complex64)
         
        # the operation matrix A for inner zone (physical domain)
        A1 = A1_spatial + A1_wavenumber
        
        return A1, A1_spatial, A1_wavenumber
        
    def build_A2(self):
        
        '''
        The operation matrix A for transition zone + boundary layer        
        '''
        
        gamma0 = self.gamma0
        nx, ny, nly = self.nx, self.ny, self.nly
        n = self.n
        
        irow_Dg, icol_Dg, val_Dg = [], [], []
        irow_HH, icol_HH, val_HH = [], [], []
        irow_OW, icol_OW, val_OW = [], [], []
        BETA, ALPHA = [], []
        
        for k in range(1, nly+1):
            
            if self.typeWeight == 'linear':
                beta_k = k/nly
            
            elif self.typeWeight == 'quadratic':
                beta_k = (k/nly)**2
            
            elif self.typeWeight == 'cube':
                beta_k = (k/nly)**3
                
            elif self.typeWeight == 'zero':
                beta_k = 0
                if k == nly:
                    beta_k = 1
                    
            elif self.typeWeight == 'one':
                beta_k = 1        
            
            else:
                raise Exception('The categories of typeWeight are:\n linear, quadratic, cube, zero, one.')            
                         
            alpha_k = 1 - beta_k
            ALPHA, BETA = np.append(ALPHA, alpha_k), np.append(BETA, beta_k)
            
            ix1, ix2 = nly - k, (nx - 1) - (nly - k)
            iy1, iy2 = nly - k, (ny - 1) - (nly - k)
            
            idx_corner, idx_l, idx_r, idx_b, idx_t = generate_index(ix1, ix2, iy1, iy2, nx)
            
            idx_side = np.concatenate((idx_l, idx_r, idx_b, idx_t))
            idx_pos = np.concatenate((idx_corner, idx_side))            
            
            #%% The linear combination of two equations: diagonal term A^{Diag}_2
            ir, ic = np.copy(idx_pos), np.copy(idx_pos)           
            v = 1*np.ones_like(idx_pos)
            irow_Dg, icol_Dg, val_Dg = np.append(irow_Dg, ir), np.append(icol_Dg, ic), np.append(val_Dg, v)
            
            #%% The weighted Helmholtz equation
            # HH eqn: off-diagonal matrix A^HH_2
            
            if self.hhorder == '2nd-central':
                const_hh = -4 + (gamma0**2)*self.epsrbg
                idx_nb = np.array([-1, 1, -nx, nx])
                coeff_nb = np.array([1, 1, 1, 1])                
            
            elif self.hhorder == '4th-central':
                
                if k == nly-1:
                    const_hh = -4 + (gamma0**2)*self.epsrbg
                    idx_nb = np.array([-1, 1, -nx, nx])
                    coeff_nb = np.array([1, 1, 1, 1])   
                else:
                    const_hh = -5 + (gamma0**2)*self.epsrbg
                    idx_nb = np.array([-1, 1, -nx, nx, -2, 2, 2*nx, -2*nx])
                    coeff_nb = np.array([16/12, 16/12, 16/12, 16/12, -1/12, -1/12, -1/12, -1/12]) 
            
            elif self.hhorder == '9point':
                const_hh = -3 + (gamma0**2)*self.epsrbg
                idx_nb = np.array([-1, 1, -nx, nx, -nx-1, -nx+1, nx-1, nx+1])
                coeff_nb = np.array([1/2, 1/2, 1/2, 1/2, 1/4, 1/4, 1/4, 1/4])            
            
            nb = len(idx_nb)
            
            if k != nly:
                # k = 1, 2, 3, ... , nly-1               
                ir = np.repeat(idx_pos, nb)
                ic = ir + np.tile(idx_nb, len(idx_pos))
                v = (alpha_k/const_hh)*np.tile(coeff_nb, len(idx_pos))
                
                irow_HH, icol_HH, val_HH = np.append(irow_HH, ir), np.append(icol_HH, ic), np.append(val_HH, v)
            
            #%% The weighted one-way wave equation
            
            # recreate arrays to easily implementation
            Idxside = idx_l, idx_r, idx_b, idx_t
            Idxcorner = np.copy(idx_corner)
            
            # OW eqn (1/2): left/right/bottom/top sides
            
            # left side
            idx_nb = (self.Idx_nb_for_side)[0, :]
            irow_OW, icol_OW, val_OW = owcoeff_oneside(self.FD_side, gamma0, beta_k, idx_l, idx_nb, irow_OW, icol_OW, val_OW)

            # right side
            idx_nb = (self.Idx_nb_for_side)[1, :]
            irow_OW, icol_OW, val_OW = owcoeff_oneside(self.FD_side, gamma0, beta_k, idx_r, idx_nb, irow_OW, icol_OW, val_OW)

            # bottom side
            idx_nb = (self.Idx_nb_for_side)[2, :]
            irow_OW, icol_OW, val_OW = owcoeff_oneside(self.FD_side, gamma0, beta_k, idx_b, idx_nb, irow_OW, icol_OW, val_OW)

            # top side
            idx_nb = (self.Idx_nb_for_side)[3, :]
            irow_OW, icol_OW, val_OW = owcoeff_oneside(self.FD_side, gamma0, beta_k, idx_t, idx_nb, irow_OW, icol_OW, val_OW)
                
            # OW eqn (2/2): corners
            
            if self.oworder == '1st-onesided':
                
                irow_OW, icol_OW, val_OW = owcoeff_oneside(self.FD_corner, gamma0, beta_k, Idxcorner[0], np.array([1, nx]), irow_OW, icol_OW, val_OW)
                irow_OW, icol_OW, val_OW = owcoeff_oneside(self.FD_corner, gamma0, beta_k, Idxcorner[1], np.array([-1, nx]), irow_OW, icol_OW, val_OW)
                irow_OW, icol_OW, val_OW = owcoeff_oneside(self.FD_corner, gamma0, beta_k, Idxcorner[2], np.array([1, -nx]), irow_OW, icol_OW, val_OW)
                irow_OW, icol_OW, val_OW = owcoeff_oneside(self.FD_corner, gamma0, beta_k, Idxcorner[3], np.array([-1, -nx]), irow_OW, icol_OW, val_OW)
                
            elif self.oworder == '2nd-onesided':
                
                irow_OW, icol_OW, val_OW = owcoeff_oneside(self.FD_corner, gamma0, beta_k, Idxcorner[0], np.array([1, 2, nx, 2*nx]), irow_OW, icol_OW, val_OW)
                irow_OW, icol_OW, val_OW = owcoeff_oneside(self.FD_corner, gamma0, beta_k, Idxcorner[1], np.array([-1, -2, nx, 2*nx]), irow_OW, icol_OW, val_OW)
                irow_OW, icol_OW, val_OW = owcoeff_oneside(self.FD_corner, gamma0, beta_k, Idxcorner[2], np.array([1, 2, -nx, -2*nx]), irow_OW, icol_OW, val_OW)
                irow_OW, icol_OW, val_OW = owcoeff_oneside(self.FD_corner, gamma0, beta_k, Idxcorner[3], np.array([-1, -2, -nx, -2*nx]), irow_OW, icol_OW, val_OW)
                
        
        irow_Dg, icol_Dg = irow_Dg.astype(int), icol_Dg.astype(int)
        irow_OW, icol_OW = irow_OW.astype(int), icol_OW.astype(int)
        
        A2_Dg = csr_matrix((val_Dg, (irow_Dg, icol_Dg)), shape=(n, n), dtype=np.complex64)
        A2_OW = csr_matrix((val_OW, (irow_OW, icol_OW)), shape=(n, n), dtype=np.complex64)
        
        A2 = A2_Dg + A2_OW
        
        if nly != 1:
        
            irow_HH, icol_HH = irow_HH.astype(int), icol_HH.astype(int)
            
            A2_HH = csr_matrix((val_HH, (irow_HH, icol_HH)), shape=(n, n), dtype=np.complex64)

            A2 += A2_HH        
        
        return A2
        
    def build_A(self):
        
        A_region1, A1_spatial, A1_wavenumber = self.build_A1()
        A_region2 = self.build_A2()        
        A = A_region1 + A_region2
        
        return A
    
def owcoeff_oneside(FD, gamma0, beta, idxside, idx_nb, irow, icol, val):
    
    ir, ic, v = oneway_stencil(FD, idxside, idx_nb, gamma0)
    irow = np.append(irow, ir)
    icol = np.append(icol, ic)
    val = np.append(val, beta*v)
    
    return irow, icol, val

def owcoeff_foursides(FD, gamma0, beta, Idxside, Idx_nb, irow, icol, val):
    
    idx_l, idx_r, idx_b, idx_t = Idxside
    
    # left side
    idx_nb = Idx_nb[0, :]
    irow, icol, val = owcoeff_oneside(FD, gamma0, beta, idx_l, idx_nb, irow, icol, val)

    # right side
    idx_nb = Idx_nb[1, :]
    irow, icol, val = owcoeff_oneside(FD, gamma0, beta, idx_r, idx_nb, irow, icol, val)

    # bottom side
    idx_nb = Idx_nb[2, :]
    irow, icol, val = owcoeff_oneside(FD, gamma0, beta, idx_b, idx_nb, irow, icol, val)

    # top side
    idx_nb = Idx_nb[3, :]
    irow, icol, val = owcoeff_oneside(FD, gamma0, beta, idx_t, idx_nb, irow, icol, val)

def round_to_int(m):
    
    if (m-int(m) >= 0.5):
        mr = np.ceil(m).astype(int)
    else:
        mr = np.floor(m).astype(int)
    
    return mr