#================================WIMPFuncs.py==================================#
# Created by Ciaran O'Hare 2019

# Description:
# This currently doesn't contain much and doesn't have much functionality,
# but it has a few functions needed to compute WIMP event rates both directional
# and non-directional. Currently only does SI interactions and modifications
# need to be made in an ad-hoc way since that wasn't the focus of the paper

# Contents:
# MinimumWIMPSpeed: computes v_min as a function of recoil energy
# MaxWIMPEnergy: computes maximum E_r for a given v_lab and v_esc
# WIMPRate: directional or non-directional WIMP event rate

#==============================================================================#



import numpy as np
from numpy import pi, sqrt, exp, zeros, size, shape
from numpy.linalg import norm
from scipy.special import erf
import LabFuncs
import Params

m_p = 0.9315*1e6
c_cm = 3.0e8*100.0 # speed of light in cm/s
GeV_2_kg = 1.0e6*1.783e-33 # convert GeV to kg

#==============================================================================#

#---------------------------------- v_min -------------------------------------#
def MinimumWIMPSpeed(E_r,A,m_chi,delta=0):
    # E_r = recoil energy in keVr
    # A = nucleus mass number
    # m_chi = Wimp mass in GeV
    # delta = inelastic scattering parameter
    m_N = m_p*A # mass of nucleus
    mu_p = 1.0e6*m_chi*m_p/(1.0e6*m_chi + m_p) # reduced proton mass
    m_N_keV = A*0.9315*1.0e6 # nucleus mass in keV
    mu_N_keV = 1.0e6*m_chi*m_N_keV/(1.0e6*m_chi + m_N_keV) # reduced nucleus mass
    v_min = sqrt(1.0/(2*m_N_keV*E_r))*(m_N_keV*E_r/mu_N_keV + delta)*3.0e8/1000.0
    return v_min

#---------------------------------- E_max -------------------------------------#
def MaxWIMPEnergy(A,v_lab,m_chi,v_esc):
    # A = nucleus mass number
    # v_lab = Lab velocity in km/s
    # m_chi = Wimp mass in GeV
    # v_esc = Escape speed in km/s
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
    # E_r = Recoil energy in keVr
    # HaloIntegral = g(vmin) or fhat(vmin,q) for non-dir. or dir. experiment
    # A = Nucleus Mass Number
    # sigma_p = SI WIMP-proton cross section in cm^2
    # m_chi = WIMP mass in GeV
    # rho_0 = Local DM density

    mu_p = 1.0e6*m_chi*m_p/(1.0e6*m_chi + m_p) # reduced mass
    FF = LabFuncs.FormFactorHelm(E_r,A)**2.0 # Form Factor^2
    R0 = (c_cm*c_cm)*((rho_0*1.0e6*A*A*sigma_p)/(2*m_chi*GeV_2_kg*mu_p*mu_p))
    HaloIntegral = HaloIntegral/(1000.0*100.0) # convert to cm^-1 s

    # Compute rate = (Rate amplitude * HaloIntegral * form factor)
    dR = R0*HaloIntegral*FF
    dR = dR*3600*24*365*1000.0 # convert to units of 1/(keVr ton year)
    return dR
