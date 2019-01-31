from numpy import pi, sqrt, exp, zeros, size, shape, linspace, meshgrid, cos, sin
from numpy import trapz
from scipy.integrate import cumtrapz
from numpy.linalg import norm
from scipy.special import erf, erfi
import LabFuncs
import Params

#================================HaloFuncs.py=================================#
# Contents:

#==============================================================================#





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
def VelocityDist_Isotropic(v,v_lab,v0=233.0,v_esc=528.0):
    sig = v0/sqrt(2.0)
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

def VelocityDist_Triaxial(v,v_lab,beta=0.9,v0=233.0,v_esc=528.0):
    sigr = sqrt(3*v0**2.0/(2.0*(3-2.0*beta)))
    sigphi = sqrt(3*v0**2.0*(1-beta)/(2.0*(3-2.0*beta)))
    sigz = sigphi
    N_esc = Nesc_Triaxial(sigr,sigphi,beta,v_esc)
    N = 1.0/((2*pi)**(1.5)*sigr*sigphi*sigz*N_esc)
    
    vr = v[:,0]
    vphi = v[:,1]
    vz = v[:,2]
    V = sqrt((vr+v_lab[0])**2.0+(vphi+v_lab[1])**2.0+(vz+v_lab[2])**2.0)
    fv3  = N*exp(-((vr+v_lab[0])**2.0/(2*sigr**2.0))\
               -((vz+v_lab[2])**2.0/(2*sigz**2.0))\
               -((vphi+v_lab[1])**2.0/(2*sigphi**2.0)))*(V<v_esc)
    return fv3

def VelocityDist_3D(v,v_lab,sig3,v0=233.0,v_esc=528.0):
    sigr = sig3[0]
    sigphi = sig3[1]
    sigz = sig3[2]
    N_esc = 1.0
    N = 1.0/((2*pi)**(1.5)*sigr*sigphi*sigz*N_esc)
    
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
def SpeedDist_Isotropic(v,v_lab,v0=233.0,v_esc=528.0):
    v_e = sqrt(sum(v_lab**2.0))
    sig = v0/sqrt(2.0)
    N_esc = Nesc_Isotropic(sig,v_esc)
    fv1 = (1.0/(N_esc*sqrt(2*pi)))*(v/(v_e*sig))\
        *(exp(-(v**2.0+v_e**2.0-2.0*v*v_e)/(2*sig**2.0))\
        -exp(-(v**2.0+v_e**2.0+2.0*v*v_e)/(2*sig**2.0)))\
        *((v)<(v_esc+v_e))
    return fv1


def SpeedDist_Triaxial(v,v_lab,beta=0.9,v0=233.0,v_esc=528.0):
    sigr=sqrt(3*v0**2.0/(2.0*(3-2.0*beta)))
    sigphi=sqrt(3*v0**2.0*(1-beta)/(2.0*(3-2.0*beta)))
    sigz=sqrt(3*v0**2.0*(1-beta)/(2.0*(3-2.0*beta)))
    N_esc = Nesc_Triaxial(sigr,sigphi,beta,v_esc)
    N = 1.0/((2*pi)**(1.5)*sigr*sigphi*sigz*N_esc)
    n = size(v)
    fv1 = zeros(shape=n)
    
    nf = 300
    costhvals = linspace(-1,1,nf)
    phivals = linspace(0,2*pi,nf)
    C,P = meshgrid(costhvals,phivals)
    for i in range(0,n):
        v1 = v[i]
        vr = v1*sqrt(1-C**2.0)*cos(P)
        vphi = v1*sqrt(1-C**2.0)*sin(P)
        vz = v1*C
        V = sqrt((vr+v_lab[0])**2.0+(vphi+v_lab[1])**2.0+(vz+v_lab[2])**2.0)

        F  = N*exp(-((vr+v_lab[0])**2.0/(2*sigr**2.0))\
                   -((vz+v_lab[2])**2.0/(2*sigz**2.0))\
                   -((vphi+v_lab[1])**2.0/(2*sigphi**2.0)))*(V<(v_esc))
        fv1[i] = (v1**2.0)*trapz(trapz(F,phivals,axis=1),costhvals)
    return fv1

def SpeedDist_3D(v,v_lab,sig3,v0=233.0,v_esc=528.0):
    sigr = sig3[0]
    sigphi = sig3[1]
    sigz = sig3[2]
    beta = 1.0-(sigr**2.0+sigz**2.0)/(2*sigr**2.0)
    #N_esc = Nesc_Triaxial(sigr,sigphi,beta,v_esc)
    N_esc = 1.0
    N = 1.0/((2*pi)**(1.5)*sigr*sigphi*sigz*N_esc)
    n = size(v)
    fv1 = zeros(shape=n)
    
    nf = 300
    costhvals = linspace(-1,1,nf)
    phivals = linspace(0,2*pi,nf)
    C,P = meshgrid(costhvals,phivals)
    for i in range(0,n):
        v1 = v[i]
        vr = v1*sqrt(1-C**2.0)*cos(P)
        vphi = v1*sqrt(1-C**2.0)*sin(P)
        vz = v1*C
        V = sqrt((vr+v_lab[0])**2.0+(vphi+v_lab[1])**2.0+(vz+v_lab[2])**2.0)

        F  = N*exp(-((vr+v_lab[0])**2.0/(2*sigr**2.0))\
                   -((vz+v_lab[2])**2.0/(2*sigz**2.0))\
                   -((vphi+v_lab[1])**2.0/(2*sigphi**2.0)))*(V<(v_esc))
        fv1[i] = (v1**2.0)*trapz(trapz(F,phivals,axis=1),costhvals)
    return fv1



#==============================================================================#
# Halo integrals
def gvmin_Isotropic(v_min,v_lab,v0=233.0,v_esc=528.0):
    # Mean inverse speed
    N_esc = Nesc_Isotropic(v0/sqrt(2.0),v_esc)
    v_e = sqrt(sum(v_lab**2.0))

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
    
    
def gvmin_Triaxial(v_min,v_lab,beta=0.9,v0=233.0,v_esc=528.0):    
    fv = SpeedDist_Triaxial(v_min,v_lab,beta=beta,v0=v0,v_esc=v_esc)
    gvmin = cumtrapz(fv,v)
    return gvmin
    
def fhat_Isotropic(v_min,x,v_lab,v0=233.0,v_esc=528.0):
    # Radon transform
    sig_v = v0/sqrt(2.0)
    N_esc = Nesc_Isotropic(sig_v,v_esc)
        
    vlabdotq = (x[:,0]*v_lab[0]+x[:,1]*v_lab[1]+x[:,2]*v_lab[2]) # recoil projection

    fhat = zeros(shape=size(vlabdotq))
    fhat[((v_min+vlabdotq)<(v_esc))] = (1/(N_esc*sqrt(2*pi*sig_v**2.0)))\
                                        *(exp(-(v_min+vlabdotq[((v_min+vlabdotq)<(v_esc))])\
                                        **2.0/(2*sig_v**2.0))\
                                        -exp(-v_esc**2.0/(2*sig_v**2.0)))
    fhat /= 2*pi
    return fhat
#==============================================================================#
