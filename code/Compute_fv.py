from numpy import *
from numpy.random import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from LabFuncs import *
from Params import *
from HaloFuncs import *
from WIMPFuncs import *
import pandas

# Set plot rc params
plt.rcParams['axes.linewidth'] = 2.5
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Halo params
HaloModel = SHMpp
v0 = HaloModel.RotationSpeed
v_esc = HaloModel.EscapeSpeed
beta = HaloModel.SausageBeta
sig_beta = HaloModel.SausageDispersionTensor
sig_iso = array([1.0,1.0,1.0])*v0/sqrt(2.0)


# Load shards
df = pandas.read_csv('../data/FitShards_red.csv')
names = df.group_id
nshards = size(names)
velocities = zeros(shape=(nshards,3))
dispersions = zeros(shape=(nshards,3))
velocities[0:(nshards),0] = df.vx # stream velocities
velocities[0:(nshards),1] = df.vy
velocities[0:(nshards),2] = df.vz
dispersions[0:(nshards),0] = df.sigx # dispersion tensors
dispersions[0:(nshards),1] = df.sigy
dispersions[0:(nshards),2] = df.sigz
pops = df.population
Psun = df.Psun
weights = ShardsWeights(names,pops,Psun)


# v_mins
n = 300
v = linspace(0.01,750.0,n)

# Times
ndays = 100
days = linspace(0.0,365.0-365.0/ndays,ndays)

# Calculate everything
fv_Iso = zeros(shape=(ndays,n))
fv_Iso_gf = zeros(shape=(ndays,n))
fv_Saus = zeros(shape=(ndays,n))
fv_Saus_gf = zeros(shape=(ndays,n))
fv_sub = zeros(shape=(ndays,n))
fv_sub_gf = zeros(shape=(ndays,n))
for i in range(0,ndays):
    fv_Iso[i,:] = SpeedDist_Triaxial(v,days[i],sig_iso)
    fv_Iso_gf[i,:] = SpeedDist_Triaxial(v,days[i],sig_iso,GravFocus=True)
    
    fv_Saus[i,:] = SpeedDist_Triaxial(v,days[i],sig_beta)
    fv_Saus_gf[i,:] = SpeedDist_Triaxial(v,days[i],sig_beta,GravFocus=True)

    for isub in range(0,nshards):
        v_s = velocities[isub,:]
        sig_s = dispersions[isub,:]
        fv_sub[i,:] += weights[isub]*SpeedDist_Triaxial(v,days[i],sig_s,v_shift=v_s)
        fv_sub_gf[i,:] += weights[isub]*SpeedDist_Triaxial(v,days[i],sig_s,v_shift=v_s,GravFocus=True)
    
    
    print('day = ',i,'of',ndays,sum(fv_sub[i,:]),sum(fv_sub_gf[i,:]))
    
    
    print('day = ',i,'of',ndays,sum(gmin_sub[i,:]),sum(gmin_sub_gf[i,:]))
    
    
savetxt('../data/fv/fv_Halo.txt',vstack((v,fv_Iso)),delimiter='\t',fmt="%1.12f")
savetxt('../data/fv/fv_Halo_GF.txt',vstack((v,fv_Iso_gf)),delimiter='\t',fmt="%1.12f")

savetxt('../data/fv/fv_Saus.txt',vstack((v,fv_Saus)),delimiter='\t',fmt="%1.12f")
savetxt('../data/fv/fv_Saus_GF.txt',vstack((v,fv_Saus_gf)),delimiter='\t',fmt="%1.12f")

savetxt('../data/fv/fv_Shards.txt',vstack((v,fv_sub)),delimiter='\t',fmt="%1.12f")
savetxt('../data/fv/fv_Shards_GF.txt',vstack((v,fv_sub_gf)),delimiter='\t',fmt="%1.12f")