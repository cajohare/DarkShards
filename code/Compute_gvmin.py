from numpy import *
from numpy.random import *
from LabFuncs import *
from Params import *
from HaloFuncs import *
from WIMPFuncs import *
import pandas

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

iS1 = 0
iS2 = arange(1,3)
iRet = arange(3,10)
iPro = arange(10,25)
iLowE = arange(25,59)


# v_mins
n = 1000
v_min = linspace(0.01,750.0,n)

# Times
ndays = 100
days = linspace(0.0,365.0-365.0/ndays,ndays)

# Calculate everything
gmin_Iso = zeros(shape=(ndays,n))
gmin_Iso_gf = zeros(shape=(ndays,n))
gmin_Saus = zeros(shape=(ndays,n))
gmin_Saus_gf = zeros(shape=(ndays,n))

gmin_S1 = zeros(shape=(ndays,n))
gmin_S1_gf = zeros(shape=(ndays,n))
gmin_S2 = zeros(shape=(ndays,n))
gmin_S2_gf = zeros(shape=(ndays,n))
gmin_Ret = zeros(shape=(ndays,n))
gmin_Ret_gf = zeros(shape=(ndays,n))
gmin_Pro = zeros(shape=(ndays,n))
gmin_Pro_gf = zeros(shape=(ndays,n))
gmin_LowE = zeros(shape=(ndays,n))
gmin_LowE_gf = zeros(shape=(ndays,n))
for i in range(0,ndays):
    gmin_Iso[i,:] = gvmin_Triaxial(v_min,days[i],sig_iso)
    gmin_Iso_gf[i,:] = gvmin_Triaxial(v_min,days[i],sig_iso,GravFocus=True)

    gmin_Saus[i,:] = gvmin_Triaxial(v_min,days[i],sig_beta)
    gmin_Saus_gf[i,:] = gvmin_Triaxial(v_min,days[i],sig_beta,GravFocus=True)

    gmin_sub = zeros(shape=(nshards,n))
    gmin_sub_gf = zeros(shape=(nshards,n))

    for isub in range(0,nshards):
        v_s = velocities[isub,:]
        sig_s = dispersions[isub,:]
        gmin_sub[isub,:] = weights[isub]*gvmin_Triaxial(v_min,days[i],sig_s,v_shift=v_s)
        gmin_sub_gf[isub,:] = weights[isub]*gvmin_Triaxial(v_min,days[i],sig_s,v_shift=v_s,GravFocus=True)

    gmin_S1[i,:] = gmin_sub[iS1,:]
    gmin_S1_gf[i,:] = gmin_sub_gf[iS1,:]

    gmin_S2[i,:] = sum(gmin_sub[iS2,:],0)
    gmin_S2_gf[i,:] = sum(gmin_sub_gf[iS2,:],0)


    gmin_Ret[i,:] = sum(gmin_sub[iRet,:],0)
    gmin_Ret_gf[i,:] = sum(gmin_sub_gf[iRet,:],0)


    gmin_Pro[i,:] = sum(gmin_sub[iPro,:],0)
    gmin_Pro_gf[i,:] = sum(gmin_sub_gf[iPro,:],0)


    gmin_LowE[i,:] = sum(gmin_sub[iLowE,:],0)
    gmin_LowE_gf[i,:] = sum(gmin_sub_gf[iLowE,:],0)

    print('day = ',i+1,'of',ndays,sum(gmin_S1[i,:]),sum(gmin_S1_gf[i,:]))


savetxt('../data/gvmin/gvmin_Halo.txt',vstack((v_min,gmin_Iso)),delimiter='\t',fmt="%1.12f")
savetxt('../data/gvmin/gvmin_Halo_GF.txt',vstack((v_min,gmin_Iso_gf)),delimiter='\t',fmt="%1.12f")

savetxt('../data/gvmin/gvmin_Saus.txt',vstack((v_min,gmin_Saus)),delimiter='\t',fmt="%1.12f")
savetxt('../data/gvmin/gvmin_Saus_GF.txt',vstack((v_min,gmin_Saus_gf)),delimiter='\t',fmt="%1.12f")

savetxt('../data/gvmin/gvmin_S1.txt',vstack((v_min,gmin_S1)),delimiter='\t',fmt="%1.12f")
savetxt('../data/gvmin/gvmin_S1_GF.txt',vstack((v_min,gmin_S1_gf)),delimiter='\t',fmt="%1.12f")

savetxt('../data/gvmin/gvmin_S2.txt',vstack((v_min,gmin_S2)),delimiter='\t',fmt="%1.12f")
savetxt('../data/gvmin/gvmin_S2_GF.txt',vstack((v_min,gmin_S2_gf)),delimiter='\t',fmt="%1.12f")

savetxt('../data/gvmin/gvmin_Ret.txt',vstack((v_min,gmin_Ret)),delimiter='\t',fmt="%1.12f")
savetxt('../data/gvmin/gvmin_Ret_GF.txt',vstack((v_min,gmin_Ret_gf)),delimiter='\t',fmt="%1.12f")

savetxt('../data/gvmin/gvmin_Pro.txt',vstack((v_min,gmin_Pro)),delimiter='\t',fmt="%1.12f")
savetxt('../data/gvmin/gvmin_Pro_GF.txt',vstack((v_min,gmin_Pro_gf)),delimiter='\t',fmt="%1.12f")

savetxt('../data/gvmin/gvmin_LowE.txt',vstack((v_min,gmin_LowE)),delimiter='\t',fmt="%1.12f")
savetxt('../data/gvmin/gvmin_LowE_GF.txt',vstack((v_min,gmin_LowE_gf)),delimiter='\t',fmt="%1.12f")
