import numpy as np
from numpy import pi, sqrt, exp, zeros, size, shape
from numpy.linalg import norm
from scipy.special import erf
import LabFuncs
import Params

#================================WIMPFuncs.f95=================================#
# Contents:

# Loading recoil distribution for WIMPs (RD_wimp)
#    WIMPRecoilDistribution: Loads RD_wimp to be used by likelihood
#    WIMPRD_Energy: Energy dependent recoil distribution
#    WIMPRD_3D: Direction dependent recoil distribution


# Energy dependnt and direction dependent rates
#    WIMPRate_Direction: Directional recoil rate d^2R(t)/dEdO
#    WIMPRate_Energy: Non-directional recoil rate dR(t)/dE
#	 MaxWIMPEnergy: Maximum recoil energy for a given date
#==============================================================================#

m_p = 0.9315*1e6
c_cm = 3.0e8*100.0 # speed of light in cm/s
GeV_2_kg = 1.0e6*1.783e-33 # convert GeV to kg

#==============================================================================#
#-------------------- Energy-Time dependent recoil rate------------------------#
def MinimumWIMPSpeed(E_r,A,m_chi):
    m_N = m_p*A # mass of nucleus
    mu_p = 1.0e6*m_chi*m_p/(1.0e6*m_chi + m_p) # reduced proton mass
    m_N_keV = A*0.9315*1.0e6 # nucleus mass in keV
    mu_N_keV = 1.0e6*m_chi*m_N_keV/(1.0e6*m_chi + m_N_keV) # reduced nucleus mass
    v_min = (sqrt(2.0*m_N_keV*E_r)/(2.0*mu_N_keV))*3.0e8/1000.0 # vmin in km/s
    return v_min

def MaxWIMPEnergy(A,v_e,m_chi):
    m_N = m_p*A
    mu_N = 1.0e6*m_N*m_chi/(1.0e6*m_chi+m_N)
    E_max_lim = 2.0*mu_N*mu_N*2.0*((v_esc+sqrt(sum(v_lab**2.0)))*1000.0/3.0e8)**2.0/m_N
    return E_max_lim

#----------------------General event rate -------------------------------------#
# Accepts either direction or non-directional energies E_r
def WIMPRate(E,t,Expt,DM,HaloModel):
    Nuc = Expt.Nucleus
    Loc = Expt.Location

    if Expt.Directional:
        ne = size(E)
        dR = zeros(shape=(ne))
        dR = dRdEdO_wimp(E,t,DM,HaloModel,Nuc,Loc)
    else:
        ne = size(E)/3
        dR = zeros(shape=(ne))
        dR = dRdE_wimp(E,t,DM,HaloModel,Nuc,Loc)
    return dR



#-------------------- Recoil  rate--------------------------------------------#
def diffRecoilRate_SI(E_r,HaloIntegral,A,sigma_p,m_chi,rho_0=0.55):
    # relevant constants
    mu_p = 1.0e6*m_chi*m_p/(1.0e6*m_chi + m_p)
    FF = LabFuncs.FormFactorHelm(E_r,A)**2.0
    R0 = (c_cm*c_cm)*((rho_0*1.0e6*A*A*sigma_p)/(2*m_chi*GeV_2_kg*mu_p*mu_p))
    HaloIntegral = HaloIntegral/(1000.0*100.0) # convert to cm^-1 s

    # Compute rate = (Rate amplitude * HaloIntegral * form factor)
    dR = R0*HaloIntegral*FF
    dR = dR*3600*24*365*1000.0 # convert to per ton-year
    return dR


