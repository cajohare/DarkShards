from numpy import pi, sqrt, exp, zeros, size, shape, linspace, meshgrid, cos, sin
from numpy import trapz, arange, array, flipud, interp
from scipy.integrate import cumtrapz
from numpy.linalg import norm
from scipy.special import erf, erfi
import LabFuncs
import Params

#================================HaloFuncs.py=================================#
# Contents:

#==============================================================================#


import healpy as hp

nside = 16
npix = 12*nside**2
dpix = 4*pi/(npix*1.0)
x_pix = zeros(shape=(npix,3))
for i in range(0,npix):
    x_pix[i,:] = hp.pix2vec(nside, i)



#==============================================================================#
# Normalisation constants
def Nesc_Isotropic(sig,v_esc):
    return erf(v_esc/(sqrt(2)*sig)) - sqrt(2.0/pi)*(v_esc/sig)*exp(-v_esc**2.0/(2.0*sig**2.0))

def Nesc_Triaxial(sigr,sigphi,beta,v_esc):
    N_esc = erf(v_esc/(sqrt(2.0)*sigr)) - sqrt((1.0-beta)/beta)\
            *exp(-v_esc**2.0/(2.0*sigphi**2.0))\
            *erfi(v_esc/(sqrt(2)*sigr)*sqrt(beta/(1-beta)))
    return N_esc
#==============================================================================#






#==============================================================================#
# Velocity distributions
def VelocityDist_Isotropic(v,day,v_LSR=233.0,sig=164.75,v_esc=528.0,\
                        v_shift=array([0.0,0.0,0.0]),GravFocus=False):
    v_lab = LabFuncs.LabVelocitySimple(day,v_LSR=v_LSR)-v_shift
   
    v0 = sig*sqrt(2.0)
    N_esc = Nesc_Isotropic(sig,v_esc)
    vr = v[:,0]
    vphi = v[:,1]
    vz = v[:,2]
    V = sqrt((vr+v_lab[0])**2.0+(vphi+v_lab[1])**2.0+(vz+v_lab[2])**2.0)
    fv3  = (1.0/(N_esc*sqrt(2*pi)*sig**3.0)*\
          exp(-((vr+v_lab[0])**2.0\
                +(vz+v_lab[2])**2.0\
                +(vphi+v_lab[1])**2.0)/(2*sig**2.0)))*(V<v_esc)
    return fv3


def VelocityDist_Triaxial(v,day,sig3,v_LSR=233.0,v_esc=528.0,\
                        v_shift=array([0.0,0.0,0.0]),GravFocus=False):
    v_lab = LabFuncs.LabVelocitySimple(day,v_LSR=v_LSR)-v_shift
    
    sigr = sig3[0]
    sigphi = sig3[1]
    sigz = sig3[2]
    beta = 1.0-(sigphi**2.0+sigz**2.0)/(2*sigr**2.0)

    if beta>0.0:
        N_esc = Nesc_Triaxial(sigr,sigphi,beta,v_esc)
    elif beta==0.0:
        N_esc = Nesc_Isotropic(sigr,v_esc)
    else:
        N_esc = 1.0
        
    N = 1.0/(N_esc*(2*pi)**(1.5)*sigr*sigphi*sigz)
    
    vr = v[:,0]
    vphi = v[:,1]
    vz = v[:,2]
    V = sqrt((vr+v_lab[0])**2.0+(vphi+v_lab[1])**2.0+(vz+v_lab[2])**2.0)
    fv3  = N*exp(-((vr+v_lab[0])**2.0/(2*sigr**2.0))\
               -((vz+v_lab[2])**2.0/(2*sigz**2.0))\
               -((vphi+v_lab[1])**2.0/(2*sigphi**2.0)))*(V<v_esc)
    return fv3
#==============================================================================#




#==============================================================================#
# Speed distributions

# Resolution of integral over velocities:
n = 51
dth = 2.0/(n-1.0)
dph = 2*pi/(2*n*1.0)
cvals = arange(-1.0,1.0,dth)
pvals = arange(0,2*pi-dph,dph)
C,P = meshgrid(cvals,pvals)
    
def SpeedDist_Isotropic(v,day,v_LSR=233.0,sig=164.75,v_esc=528.0,\
                        v_shift=array([0.0,0.0,0.0]),GravFocus=False):
    v_lab = LabFuncs.LabVelocitySimple(day,v_LSR=v_LSR)-v_shift
    v_e = sqrt(sum(v_lab**2.0))
    v0 = sig*sqrt(2.0)
    N_esc = Nesc_Isotropic(sig,v_esc)
    fv1 = (1.0/(N_esc*sqrt(2*pi)))*(v/(v_e*sig))\
        *(exp(-(v**2.0+v_e**2.0-2.0*v*v_e)/(2*sig**2.0))\
        -exp(-(v**2.0+v_e**2.0+2.0*v*v_e)/(2*sig**2.0)))\
        *((v)<(v_esc+v_e))
    fv1 /= trapz(fv1,v)
        
    if GravFocus:
        nvals = size(v)
        fvJ = zeros(shape=nvals)
        for i in range(0,nvals):
            vv = v[i]
            voff2 = (vv*sqrt(1-C**2)*cos(P)+v_lab[0])**2\
                    + (vv*sqrt(1-C**2.0)*sin(P)+v_lab[1])**2.0\
                    + (vv*C+v_lab[2])**2
            fv3 = vv**2.0*(1.0/(sqrt(pi)*pi))*(1.0/v0**3)*exp(-voff2/v0**2)
            fJ = fv3*LabFuncs.GravFocusAngles(vv,C,P,day,sig=sig)
            fvJ[i] = dth*dph*sum(sum(fJ))
        fv1 += fvJ
    
    return fv1



def SpeedDist_Triaxial(v,day,sig3,v_LSR=233.0,v_esc=528.0,\
                        v_shift=array([0.0,0.0,0.0]),GravFocus=False,GalFrame=False):
    sigr = sig3[0]
    sigphi = sig3[1]
    sigz = sig3[2]

    beta = 1.0-(sigphi**2.0+sigz**2.0)/(2*sigr**2.0)
    if beta>0.0:
        N_esc = Nesc_Triaxial(sigr,sigphi,beta,v_esc)
    elif beta==0.0:
        N_esc = Nesc_Isotropic(sigr,v_esc)
    else:
        N_esc = 1.0
   

    N = 1.0/(N_esc*(2*pi)**(1.5)*sigr*sigphi*sigz)
    n = size(v)
    fv1 = zeros(shape=n)

    if GravFocus==False:
        if GalFrame:
            v_off = -v_shift
            v_max = v_esc
        else:
            v_e = LabFuncs.LabVelocitySimple(day,v_LSR=v_LSR)
            v_max = v_esc+sqrt(sum(v_e**2.0))
            v_off = v_e-v_shift
        
        for i in range(0,n):
            v1 = v[i]
            vr = v1*sqrt(1-C**2.0)*cos(P)+v_off[0]
            vphi = v1*sqrt(1-C**2.0)*sin(P)+v_off[1]
            vz = v1*C+v_off[2]
            V = sqrt(vr**2.0+vphi**2.0+vz**2.0)
            
            F  = N*exp(-(vr**2.0/(2*sigr**2.0))\
                       -(vz**2.0/(2*sigz**2.0))\
                       -(vphi**2.0/(2*sigphi**2.0)))*(V<v_max)
            fv1[i] = (v1**2.0)*dth*dph*sum(sum(F))
        fv1[v>v_max] = 0.0
        fv1 /= trapz(fv1,v)
    else:
        
        v_off = LabFuncs.v_pec+array([0.0,v_LSR,0.0])-v_shift
        vv_e = sqrt(sum(v_off**2.0))

        for i in range(0,n):
            v1 = v[i]
            vr,vphi,vz = LabFuncs.v_infinity(v1,C,P,day)

            vr += v_off[0]
            vphi += v_off[1]
            vz += v_off[2]
            V = sqrt(vr**2.0+vphi**2.0+vz**2.0)
            F  = N*exp(-(vr**2.0/(2*sigr**2.0))\
                       -(vz**2.0/(2*sigz**2.0))\
                       -(vphi**2.0/(2*sigphi**2.0)))*(V<(v_esc+vv_e))
            fv1[i] = (v1**2.0)*dth*dph*sum(sum(F))
            
        fv1[v>(v_esc+vv_e)] = 0.0
    return fv1




def SpeedDist_Triaxial_alt(v,day,sig3,v_LSR=233.0,v_esc=528.0,\
                        v_shift=array([0.0,0.0,0.0]),GravFocus=False):
    sigr = sig3[0]
    sigphi = sig3[1]
    sigz = sig3[2]

    beta = 1.0-(sigphi**2.0+sigz**2.0)/(2*sigr**2.0)
    if beta>0.0:
        N_esc = Nesc_Triaxial(sigr,sigphi,beta,v_esc)
    elif beta==0.0:
        N_esc = Nesc_Isotropic(sigr,v_esc)
    else:
        N_esc = 1.0
    
    v_e = LabFuncs.LabVelocitySimple(day,v_LSR=v_LSR)

    N = 1.0/(N_esc*(2*pi)**(1.5)*sigr*sigphi*sigz)
    n = size(v)
    fv1 = zeros(shape=n)
    
    if GravFocus==False:
        v_off = v_e-v_shift
        vv_e = sqrt(sum(v_off**2.0))
        
        for i in range(0,n):
            v1 = v[i]
            vr = v1*x_pix[:,0]+v_off[0]
            vphi = v1*x_pix[:,1]+v_off[1]
            vz = v1*x_pix[:,2]+v_off[2]
            V = sqrt(vr**2.0+vphi**2.0+vz**2.0)

            F  = N*exp(-(vr**2.0/(2*sigr**2.0))\
                       -(vz**2.0/(2*sigz**2.0))\
                       -(vphi**2.0/(2*sigphi**2.0)))*(V<(v_esc+vv_e))
            fv1[i] = (v1**2.0)*sum(F)*dpix
    else:
        
        v_off = LabFuncs.v_pec+array([0.0,v_LSR,0.0])-v_shift
        vv_e = sqrt(sum(v_off**2.0))

        for i in range(0,n):
            v1 = v[i]
            vr,vphi,vz = LabFuncs.v_infinity_alt(v1*x_pix,day)

            vr += v_off[0]
            vphi += v_off[1]
            vz += v_off[2]
            V = sqrt(vr**2.0+vphi**2.0+vz**2.0)
            F  = N*exp(-(vr**2.0/(2*sigr**2.0))\
                       -(vz**2.0/(2*sigz**2.0))\
                       -(vphi**2.0/(2*sigphi**2.0)))*(V<(v_esc+vv_e))
            fv1[i] = (v1**2.0)*sum(F)*dpix
            
    fv1[v>(v_esc+vv_e)] = 0.0
    return fv1




#==============================================================================#
# Halo integrals
def gvmin_Isotropic(v_min,day,v_LSR=233.0,sig=164.75,v_esc=528.0,\
                    v_shift=array([0.0,0.0,0.0]),GravFocus=False,v_exponent=-1.0):
   
    if (GravFocus) or (v_exponent!=-1.0):
        
        v_min_fine = linspace(0.0001,800.0,300)
        fv = flipud((v_min_fine**v_exponent)*\
                    SpeedDist_Isotropic(v_min_fine,day,v_LSR=v_LSR,sig=sig,\
                                 v_esc=v_esc,v_shift=v_shift,GravFocus=GravFocus))
        gvmin_fine = zeros(shape=size(v_min_fine))
        gvmin_fine[0:-1] = flipud(cumtrapz(fv,v_min_fine))
        gvmin_fine[-1] = 0.0
        gvmin = interp(v_min,v_min_fine,gvmin_fine)
        
    else:
        # If Grav focus being ignored and exponent=-1, can use analytic result
        v_lab = LabFuncs.LabVelocitySimple(day,v_LSR=v_LSR)-v_shift
        N_esc = Nesc_Isotropic(sig,v_esc)
        v_e = sqrt(sum(v_lab**2.0))
        v0 = sig*sqrt(2.0)
        x = v_min/v0
        z = v_esc/v0
        y = v_e/v0

        gvmin = zeros(shape=shape(v_min))
        g1 = (1.0/(v0*y))
        g2 = (1.0/(2.0*N_esc*v0*y))*(erf(x+y)-erf(x-y)-(4.0/sqrt(pi))*y*exp(-z**2))
        g3 = (1.0/(2.0*N_esc*v0*y))*(erf(z)-erf(x-y)-(2.0/sqrt(pi))*(y+z-x)*exp(-z**2))
        gvmin[(x<abs(y-z))&(z<y)] = g1
        gvmin[(x<abs(y-z))&(z>y)] = g2[(x<abs(y-z))&(z>y)]
        gvmin[(abs(y-z)<x)&(x<(y+z))] = g3[(abs(y-z)<x)&(x<(y+z))]
        gvmin[(y+z)<x] = 0.0
    
    return gvmin
    
    
def gvmin_Triaxial(v_min,day,sig,v_LSR=233.0,v_esc=528.0,\
                  v_shift=array([0.0,0.0,0.0]),GravFocus=False,v_exponent=-1.0):    
    v_min_fine = linspace(0.0001,800.0,300)
    fv = flipud((v_min_fine**v_exponent)*\
                SpeedDist_Triaxial(v_min_fine,day,sig,v_LSR=v_LSR,v_esc=v_esc,\
                                              v_shift=v_shift,GravFocus=GravFocus))
    gvmin_fine = zeros(shape=size(v_min_fine))
    gvmin_fine[0:-1] = flipud(cumtrapz(fv,v_min_fine))
    gvmin_fine[-1] = 0.0
    gvmin = interp(v_min,v_min_fine,gvmin_fine)
    return gvmin
    
def fhat_Isotropic(v_min,x,day,v_LSR=233.0,sig=164.75,v_esc=528.0,\
                    v_shift=array([0.0,0.0,0.0])):
    # RECOIL VECTOR x JUST NEEDS TO BE IN SAME COORDINATES AS v_lab
    v_lab = LabFuncs.LabVelocitySimple(day,v_LSR=v_LSR)-v_shift

    v0 = sig*sqrt(2.0)
    N_esc = Nesc_Isotropic(sig,v_esc)
        
    vlabdotq = (x[:,0]*v_lab[0]+x[:,1]*v_lab[1]+x[:,2]*v_lab[2]) # recoil projection

    fhat = zeros(shape=size(vlabdotq))
    fhat[((v_min+vlabdotq)<(v_esc))] = (1/(N_esc*sqrt(2*pi*sig**2.0)))\
                                        *(exp(-(v_min+vlabdotq[((v_min+vlabdotq)<(v_esc))])\
                                        **2.0/(2*sig**2.0))\
                                        -exp(-v_esc**2.0/(2*sig**2.0)))
    fhat /= 2*pi
    return fhat


def fhat_Triaxial(v_min,x,day,sig3,v_LSR=233.0,v_esc=528.0,\
                    v_shift=array([0.0,0.0,0.0])):
    # RECOIL VECTOR x MUST BE IN GALACTIC COORDINATES
    
    v_lab = LabFuncs.LabVelocitySimple(day,v_LSR=v_LSR)-v_shift
    
    sigr = sig3[0]
    sigphi = sig3[1]
    sigz = sig3[2]

    beta = 1.0-(sigphi**2.0+sigz**2.0)/(2*sigr**2.0)
    if beta>0.0:
        N_esc = Nesc_Triaxial(sigr,sigphi,beta,v_esc)
    else:
        N_esc = 1.0
                
    vlabdotq = (x[:,0]*v_lab[0]+x[:,1]*v_lab[1]+x[:,2]*v_lab[2]) # recoil projection
    
    qsq = (x[:,0]*sigr)**2.0 + (x[:,1]*sigphi)**2.0 + (x[:,2]*sigz)**2.0
    
    fhat = zeros(shape=size(vlabdotq))
    mask = ((v_min+vlabdotq)<(v_esc))
    fhat[mask] = (1.0/(N_esc*sqrt(2*pi*qsq[mask])))*\
              (exp(-(v_min+vlabdotq[mask])**2.0/(2*qsq[mask]))-\
              exp(-v_esc**2.0/(2*qsq[mask])))

    fhat /= 2*pi
    return fhat


#==============================================================================#
