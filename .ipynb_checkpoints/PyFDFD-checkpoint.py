import numpy as np
from scipy.sparse import linalg, diags, spmatrix

from matrix import OperationMatrix
from LS_computation import compute_G, compute_mask2D

def round_to_int(m):
       
    if (m-int(m) >= 0.5):
        mr = np.ceil(m).astype(int)
    else:
        mr = np.floor(m).astype(int)
    
    return mr

def average_matrix(A):
    
    ny, nx = A.shape    
    A1 = np.zeros([ny-1, nx-1], dtype=np.complex64)
    
    for ir in range(ny-1):
        for ic in range(nx-1):            
            A1[ir, ic] = A[ir, ic] + A[ir, ic+1] + A[ir+1, ic] + A[ir+1, ic+1]
            
    A1 = 0.25*A1
    
    return A1
    
class simulation:
    
    '''
    The simulation is a hybrid model that there are two zones:
    - Inner zone is governed by the Helmholtz equation,
    - Outer zone is modelled by linear combination of two equations
      (the Helmholtz equation and the one-way wave equation).    
    '''
    
    def __init__(self, domain, gs, epsrbg=1.0):
        
        self.xHHmin, self.xHHmax, self.yHHmin, self.yHHmax = domain
        self.gs = gs
        self.epsrbg = epsrbg
        
        # number of the grid points in x- and y-directions
        self.nxHH = round_to_int((self.xHHmax-self.xHHmin)/self.gs) + 1
        self.nyHH = round_to_int((self.yHHmax-self.yHHmin)/self.gs) + 1
        
        # position of the grid points inside the physical domain
        self.xHH = np.linspace(self.xHHmin, self.xHHmax, self.nxHH)
        self.yHH = np.linspace(self.yHHmin, self.yHHmax, self.nyHH)
        self.XHH, self.YHH = np.meshgrid(self.xHH, self.yHH)
        
        # initialize the relative permittivty distribution inside the physical domain 
        self.MatHH2D = self.epsrbg*np.ones([self.nyHH, self.nxHH], dtype=np.complex64)
        
        # background medium (half-integer grid --> '_' at the end of variable name)
#         self.Mat2D_ = self.epsrbg*np.ones([self.ny+1, self.nx+1], dtype=np.complex64)
#         self.Mat2D = self.epsrbg*np.ones([self.ny, self.nx], dtype=np.complex64)
        
    def domain(self, xHHmin, xHHmax, yHHmin, yHHmax, t):
        
        self.t = t
        
        self.nly = round_to_int(t/self.gs)
                
        self.nx = self.nxHH + 2*self.nly
        self.ny = self.nyHH + 2*self.nly
        self.xmin, self.xmax = xHHmin - t, xHHmax + t
        self.ymin, self.ymax = yHHmin - t, yHHmax + t
        
        self.x = np.linspace(self.xmin, self.xmax, self.nx)
        self.y = np.linspace(self.ymin, self.ymax, self.ny)
                
        self.X, self.Y = np.meshgrid(self.x, self.y)
    
    def insert_object(self, ObjectHH2D):

        if ObjectHH2D.shape != (self.nyHH, self.nxHH):
            raise Exception(f'The size of ObjectHH2D does not match to the physical domain ({self.nyHH},{self.nxHH})') 
        
        self.Mat2D[self.nly:-self.nly, self.nly:-self.nly] = ObjectHH2D
        
    def objectinfo(self, typeObject, geometry, epsrmat):   
        
        if typeObject == 'circle':
            
            # build an curved object with smooth pixel 
            xc, yc, radius = geometry
            
            xHHmat = np.linspace(self.xHHmin-0.5*self.gs, self.xHHmax+0.5*self.gs, self.nxHH+1)
            yHHmat = np.linspace(self.yHHmin-0.5*self.gs, self.yHHmax+0.5*self.gs, self.nyHH+1)

            XHHmat, YHHmat = np.meshgrid(xHHmat, yHHmat)
            XHHmatf, YHHmatf = XHHmat.flatten(), YHHmat.flatten()
            Dist1D = np.sqrt( (XHHmatf-xc)**2 + (YHHmatf-yc)**2 )
            
            MatHH2D_ = self.epsrbg*np.ones([self.nyHH+1, self.nxHH+1], dtype=np.complex64)
            MatHH1D_ = MatHH2D_.flatten()            
            MatHH1D_[Dist1D<=radius] = epsrmat
            MatHH2D_ = MatHH1D_.reshape(self.nyHH+1, self.nxHH+1)
            
            MatHH2D_circle = average_matrix(MatHH2D_)
            
            # add the circle object to "MatHH2D" array
            MatHH1D_circle = MatHH2D_circle.flatten()
            index_circle = np.where( MatHH1D_circle!=self.epsrbg )
            
            MatHH1D = (np.copy(self.MatHH2D)).flatten()
            MatHH1D[index_circle] = MatHH1D_circle[index_circle]
            
            self.MatHH2D = MatHH1D.reshape(self.nyHH, self.nxHH)

        elif typeObject == 'rectangle':
            
            # build an rectangular object
            x1, x2, y1, y2 = geometry
            
            ix1 = round_to_int((x1-self.xmin)/self.gs)
            ix2 = round_to_int((x2-self.xmin)/self.gs)
            iy1 = round_to_int((y1-self.ymin)/self.gs)
            iy2 = round_to_int((y2-self.ymin)/self.gs)
            
            # add the rectangular object to "MatHH2D" array
            (self.Mat2D)[iy1:iy2+1, ix1:ix2+1] = epsrmat
            self.MatHH2D = (self.Mat2D)[nly:-nly, nly:-nly]
            
        else:
            raise Exception('TypeObject must be either:\n1: circle\n2: rectangle.')
            
    def source(self, incparams):
        
        self.typeSource = incparams[0]        
        
        if self.typeSource == 'PW':
            
            _, freq, amp, xsrc, ysrc, thetadeg = incparams
            
            self.freq = freq
            self.amp = amp
            self.xsrc, self.ysrc = xsrc, ysrc
            self.theta = thetadeg*np.pi/180
            
            C = 299792458 # [m/s]
            self.k0 = 2*np.pi*freq/C
            self.wlen = freq/C
            self.gamma0 = self.k0*self.gs
            kx = self.k0*np.cos(self.theta)
            ky = self.k0*np.sin(self.theta)
            
            term1 = np.exp(1j*kx*(self.XHH-self.xsrc))
            term2 = np.exp(1j*ky*(self.YHH-self.ysrc))
            self.UincHH2D = amp*term1*term2
            
            self.bHH2D = -(self.gamma0**2)*(self.MatHH2D-self.epsrbg)*self.UincHH2D
        
        elif self.typeSource == 'PTS':
            
            _, freq, amp, xsrc, ysrc = incparams
            
            if xsrc < self.xHHmin or xsrc > self.xHHmax or ysrc < self.yHHmin or ysrc > self.yHHmax:
                raise Exception('The position of point source is out of the physical domain.\n')
            
            bHH2D = np.zeros_like(self.XHH, dtype=np.complex64)
            
            ixsrc = np.where(np.abs(xsrc-self.xHH)<=0.5*self.gs)[0][0]
            iysrc = np.where(np.abs(ysrc-self.yHH)<=0.5*self.gs)[0][0]
            
            self.bHH2D[iysrc, ixsrc] = amp    
            self.UincHH2D = np.copy(self.bHH2D)
        
        else:
            raise Exception('typeSource must be either:\n1: PW (plane wave)\n2: PTS (single point source)')
    
    def boundarycondtion(self, owparam):
        
        '''
        in this code, typeOWeqn can be either 'MUR1' or 'MUR2'.
        '''
        self.typeOWeqn, self.nly, self.typeWeight = owparam
        
        
    def fdscheme(self, hhorder, oworder):
        self.hhorder, self.oworder = hhorder, oworder
            
#     def hhsolver(self, orderHH, orderOW, typeWeight, typeK, typeDamp, sigma_0): # old version
    def hhsolver(self):
        
        # map the information from the physical domain (nyHH, nxHH) to computational domain (ny, nx)  
        self.Mat2D = self.epsrbg*np.ones([self.ny, self.nx], dtype=np.complex64)
        self.Mat2D[self.nly:-self.nly, self.nly:-self.nly] = self.MatHH2D
        
        b2D = np.zeros_like(self.Mat2D)
        b2D[self.nly:-self.nly, self.nly:-self.nly] = self.bHH2D
        self.b = b2D.flatten()
        
        # build the operation matrix A
        matrix = OperationMatrix(self.nxHH, self.nyHH, self.nly)
#         matrix.configuration(self.epsrbg, self.gamma0, orderHH, orderOW, typeWeight, typeK, typeDamp, sigma_0) # old version
        matrix.configuration(self.epsrbg, self.gamma0, self.typeOWeqn, self.hhorder, self.oworder, self.typeWeight)
        matrix.medium(self.Mat2D)
        self.A = matrix.build_A()
        
        # solve for the scattered field entire the computational domain
        x1D = linalg.spsolve(self.A, self.b)
        x2D = x1D.reshape([self.ny, self.nx])
        
        self.Uscat2D = np.copy(x2D)
        
        if self.typesrc == 'PTS':
            self.Utot2D = np.copy(x2D)
            
        else:
            self.Uinc2D = np.zeros_like(self.Mat2D)
            self.Uinc2D[self.nly:-self.nly, self.nly:-self.nly] = self.UincHH2D
            self.Utot2D = x2D + self.Uinc2D
            
        self.UscatHH2D = x2D[self.nly:-self.nly, self.nly:-self.nly]
        self.UtotHH2D = self.Utot2D[self.nly:-self.nly, self.nly:-self.nly]
                
        return self.UincHH2D, self.UscatHH2D, self.UtotHH2D
   
        
    def viz(self, typefield, domain_box):
        
        x1, x2, y1, y2 = domain_box
        
        ix1 = round_to_int((x1-self.xmin)/self.gs)
        ix2 = round_to_int((x2-self.xmin)/self.gs)
        iy1 = round_to_int((y1-self.ymin)/self.gs)
        iy2 = round_to_int((y2-self.ymin)/self.gs)
        
        if typefield == 'sc':            
            Field2D = np.copy(self.Uscat2D)   
            
        elif typefield == 'tot':            
            Field2D = np.copy(self.Utot2D)
            
        Field2D_viz = Field2D[iy1:iy2+1, ix1:ix2+1]
        
        return Field2D_viz