from numpy import *
from numpy.random import *
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
#from mpl_toolkits.basemap import Basemap
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
from matplotlib import colors
import pandas
from scipy.spatial import Delaunay
from scipy.spatial import ConvexHull
from scipy.stats import zscore,chi2,multivariate_normal
from sklearn import mixture
from scipy.special import erfinv
from scipy.stats import gaussian_kde
import cartopy.crs as ccrs
import matplotlib.ticker as mticker

# Galpy
from galpy.orbit import Orbit
from galpy.potential import MWPotential2014
from mpl_toolkits.mplot3d import Axes3D
from astropy import units
from skimage import measure

Sun = array([8.122,0.0,0.005])
Sun_cyl = array([Sun[0],0.0,Sun[2]])

from astropy import units as u
from astropy.coordinates import SkyCoord, get_constellation

# cygnus_stars = array(['β','η','γ','α','γ','δ','ι','κ','ι','δ','γ','ε','ζ'])
# #cygnus_stars = ['Deneb','gamma cyg']
# nst = size(cygnus_stars)
# cyg = zeros(shape=(nst,2))
# for i in range(0,nst):
#     c = SkyCoord.from_name(cygnus_stars[i]+' Cyg').galactic
#     cyg[i,:] = array([c.l.degree,c.b.degree])

cyg = array([[ 62.10963941,   4.57150891],
       [ 71.01544763,   3.3646167 ],
       [ 78.14859103,   1.86708845],
       [ 84.28473664,   1.99754612],
       [ 78.14859103,   1.86708845],
       [ 78.70955616,  10.24302209],
       [ 83.61190613,  15.44876931],
       [ 84.40177176,  17.85322722],
       [ 83.61190613,  15.44876931],
       [ 78.70955616,  10.24302209],
       [ 78.14859103,   1.86708845],
       [ 75.95136158,  -5.71541249],
       [ 76.75381354, -12.45226928]])

def reverse_colourmap(cmap, name = 'my_cmap_r'):
    """
    In: 
    cmap, name 
    Out:
    my_cmap_r

    Explanation:
    t[0] goes from 0 to 1
    row i:   x  y0  y1 -> t[0] t[1] t[2]
                   /
                  /
    row i+1: x  y0  y1 -> t[n] t[1] t[2]

    so the inverse should do the same:
    row i+1: x  y1  y0 -> 1-t[0] t[2] t[1]
                   /
                  /
    row i:   x  y1  y0 -> 1-t[n] t[2] t[1]
    """        
    reverse = []
    k = []   

    for key in cmap._segmentdata:    
        k.append(key)
        channel = cmap._segmentdata[key]
        data = []

        for t in channel:                    
            data.append((1-t[0],t[2],t[1]))            
        reverse.append(sorted(data))    

    LinearL = dict(zip(k,reverse))
    my_cmap_r = mpl.colors.LinearSegmentedColormap(name, LinearL) 
    return my_cmap_r



def MySquarePlot(xlab='',ylab='',\
                 lw=2.5,lfs=45,tfs=25,size_x=13,size_y=12,Grid=False):
    plt.rcParams['axes.linewidth'] = lw
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif',size=tfs)
    
    fig = plt.figure(figsize=(size_x,size_y))
    ax = fig.add_subplot(111)
    
    ax.set_xlabel(xlab,fontsize=lfs)
    ax.set_ylabel(ylab,fontsize=lfs) 
    
    ax.tick_params(which='major',direction='in',width=2,length=13,right=True,top=True,pad=7)
    ax.tick_params(which='minor',direction='in',width=1,length=10,right=True,top=True)
    if Grid:
        ax.grid()
    return fig,ax

def MyDoublePlot(xlab1='',ylab1='',xlab2='',ylab2='',\
                 wspace=0.25,lw=2.5,lfs=45,tfs=25,size_x=20,size_y=11,Grid=False):
    plt.rcParams['axes.linewidth'] = lw
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif',size=tfs)
    
    fig, axarr = plt.subplots(1, 2,figsize=(size_x,size_y))
    gs = gridspec.GridSpec(1, 2)
    gs.update(wspace=wspace)
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax1.tick_params(which='major',direction='in',width=2,length=13,right=True,top=True,pad=7)
    ax1.tick_params(which='minor',direction='in',width=1,length=10,right=True,top=True)
    ax2.tick_params(which='major',direction='in',width=2,length=13,right=True,top=True,pad=7)
    ax2.tick_params(which='minor',direction='in',width=1,length=10,right=True,top=True)
    
    ax1.set_xlabel(xlab1,fontsize=lfs)
    ax1.set_ylabel(ylab1,fontsize=lfs) 
    
    ax2.set_xlabel(xlab2,fontsize=lfs)
    ax2.set_ylabel(ylab2,fontsize=lfs) 
    
    if Grid:
        ax1.grid()
        ax2.grid()
    return fig,ax1,ax2


def MyTriplePlot(xlab1='',ylab1='',xlab2='',ylab2='',xlab3='',ylab3='',\
                 wspace=0.25,lw=2.5,lfs=45,tfs=25,size_x=20,size_y=7,Grid=False):
    plt.rcParams['axes.linewidth'] = lw
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif',size=tfs)
    
    fig, axarr = plt.subplots(1, 3,figsize=(size_x,size_y))
    gs = gridspec.GridSpec(1, 3)
    gs.update(wspace=wspace)
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax3 = plt.subplot(gs[2])
    
    ax1.tick_params(which='major',direction='in',width=2,length=13,right=True,top=True,pad=7)
    ax1.tick_params(which='minor',direction='in',width=1,length=10,right=True,top=True)
    
    ax2.tick_params(which='major',direction='in',width=2,length=13,right=True,top=True,pad=7)
    ax2.tick_params(which='minor',direction='in',width=1,length=10,right=True,top=True)
    
    ax3.tick_params(which='major',direction='in',width=2,length=13,right=True,top=True,pad=7)
    ax3.tick_params(which='minor',direction='in',width=1,length=10,right=True,top=True)
    
    ax1.set_xlabel(xlab1,fontsize=lfs)
    ax1.set_ylabel(ylab1,fontsize=lfs) 
    
    ax2.set_xlabel(xlab2,fontsize=lfs)
    ax2.set_ylabel(ylab2,fontsize=lfs) 
    
    ax3.set_xlabel(xlab3,fontsize=lfs)
    ax3.set_ylabel(ylab3,fontsize=lfs) 
    
    if Grid:
        ax1.grid()
        ax2.grid()
        ax3.grid()
    return fig,ax1,ax2,ax3



def MollweideMap(ax,TH,PH,fv0,cmin,cmax,nlevels,cmap,tfs,PlotCygnus=False,gridlinecolor='k',GalacticPlane=False):
    plt.rcParams['axes.linewidth'] = 3
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif',size=15)

    
    ax.contourf(rad2deg(PH), rad2deg(TH), fv0,nlevels,transform=ccrs.PlateCarree(),cmap=cmap,vmin=cmin,vmax=cmax)
    gl = ax.gridlines(color=gridlinecolor,linewidth=1.5, linestyle='--',alpha=0.5)
    gl.ylocator = mticker.FixedLocator([-90,-60, -30, 0, 30, 60,90])
    ax.outline_patch.set_linewidth(3)
   

    tx = array([r'$-60^\circ$',r'$-30^\circ$',r'$0^\circ$',r'$+30^\circ$',r'$+60^\circ$']) 
    xtx = array([0.17,0.05,-0.01,0.05,0.18])
    ytx = array([0.08,0.26,0.49,0.72,0.9])
    
    for i in range(0,size(xtx)):
        plt.text(xtx[i],ytx[i],tx[i],transform=ax.transAxes,horizontalalignment='right',verticalalignment='center',fontsize=tfs)


    if PlotCygnus==True:
        ax.plot(-cyg[0:4,0],cyg[0:4,1],'-',color='crimson',transform=ccrs.PlateCarree())
        ax.plot(-cyg[4:,0],cyg[4:,1],'-',color='crimson',transform=ccrs.PlateCarree())
        ax.plot(-cyg[:,0],cyg[:,1],'.',color='k',ms=5,transform=ccrs.PlateCarree())

    if GalacticPlane==True:
        ax.plot([-181,181],[0,0],'-',color=gridlinecolor,lw=1.5,transform=ccrs.PlateCarree())
        ax.text(125,4,'Galactic',color=gridlinecolor,transform=ccrs.PlateCarree(),fontsize=int(tfs*0.8))
        ax.text(135,-10,'plane',color=gridlinecolor,transform=ccrs.PlateCarree(),fontsize=int(tfs*0.8))
    return





def PointScatter(xin,yin):
    dens = gaussian_kde(vstack([xin,yin]))(vstack([xin,yin]))
    idx = dens.argsort()
    x, y, dens = xin[idx], yin[idx], dens[idx]
    return x,y,dens


def in_hull(p, hull):
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p)>=0

def chaikins_corner_cutting(x_edge,y_edge, refinements=3):
    edge=zeros(shape=(size(x_edge)+1,2))
    edge[:-1,0] = x_edge
    edge[:-1,1] = y_edge
    edge[-1,0] = x_edge[0]
    edge[-1,1] = y_edge[0]
    coords = array(edge)
    for _ in range(refinements):
        L = coords.repeat(2, axis=0)
        R = empty_like(L)
        R[0] = L[0]
        R[2::2] = L[1:-1:2]
        R[1:-1:2] = L[2::2]
        R[-1] = L[-1]
        coords = L * 0.75 + R * 0.25

    return coords

def col_alpha(col,alpha=0.1):
    rgb = colors.colorConverter.to_rgb(col)
    bg_rgb = [1,1,1]
    return [alpha * c1 + (1 - alpha) * c2
            for (c1, c2) in zip(rgb, bg_rgb)]


def RemovePhaseSpaceOutliers(x,y,z,U,V,W,feh,z_th=6.0):
    # reduced points
    nstars = size(x)
    if nstars>5:
        ZSCORE = abs(zscore(z))+abs(zscore(x))+abs(zscore(y))
    else:
        ZSCORE = (z_th-1)*ones(shape=nstars)    
    x_red = x[ZSCORE<z_th]
    y_red = y[ZSCORE<z_th]
    z_red = z[ZSCORE<z_th]
    U_red = U[ZSCORE<z_th]
    V_red = V[ZSCORE<z_th]
    W_red = W[ZSCORE<z_th]
    feh_red = feh[ZSCORE<z_th]
    return x_red,y_red,z_red,U_red,V_red,W_red,feh_red



def SunProb(x_meens,x_covs):
    ng = shape(x_meens)[0]
    Psun = zeros(shape=ng)
    for i in range(0,ng):
        xyz = squeeze(x_meens[i,:])
        dxyz = squeeze(x_covs[i,:,:])
        Lmax = log(multivariate_normal.pdf(xyz, mean=xyz, cov=dxyz))
        Lsun = log(multivariate_normal.pdf(Sun, mean=xyz, cov=dxyz))
        dL = -2*(Lsun-Lmax)
        Psun[i] = sqrt(2)*erfinv(chi2.cdf(dL,3))
    
    if sum(Psun)==inf:
        for i in range(0,ng):
            xyz = squeeze(x_meens[i,:])
            dxyz = squeeze(x_covs[i,:,:])
            dxyz = diagonal(dxyz)
            Lmax = log(multivariate_normal.pdf(xyz, mean=xyz, cov=dxyz))
            Lsun = log(multivariate_normal.pdf(Sun, mean=xyz, cov=dxyz))
            dL = -2*(Lsun-Lmax)
            Psun[i] = sqrt(2)*erfinv(chi2.cdf(dL,3))
    Psun = squeeze(Psun)
    return Psun
    
    
def fv_1D(vfine,clf,i):
    covs = clf.covariances_
    meens = clf.means_
    ws = clf.weights_

    fv = zeros(shape=shape(vfine))
    if ndim(covs)>2:
        for k in range(0,shape(covs)[0]):
            U = squeeze(linalg.inv(covs[k,:,:]))
            U0 = U[i,i]
            V = U[i,:]    
            V = delete(V, i, axis=0)
            W = delete(U, i, axis=0)
            W = delete(W, i, axis=1)
            U = U0 - linalg.multi_dot([V, linalg.inv(W), V.T])
            
            v0 = meens[k,i]
            #Norm = (1.0/sqrt(2*pi))*sqrt(linalg.det(W))
            Norm = 1.0
            fv += ws[k]*Norm*exp(-0.5*(vfine-v0)*U*(vfine-v0))
            #if (shape(covs)[0])==2:
            #    print k,U,U0,linalg.multi_dot([V, linalg.inv(W), V.T])
            #    print '----'
    else:
        # If diagonal just use normal formula
        sig0_sq = covs[0,i]
        v0 = meens[0,i]
        Norm = (1.0/sqrt(2*pi*sig0_sq))
        fv = Norm*exp(-(vfine-v0)**2.0/(2*sig0_sq))
    fv /= trapz(fv,vfine)
    return fv


def fv_2D(V1,V2,clf,i,j):
    # see http://www-biba.inrialpes.fr/Jaynes/cappe1.pdf
    covs = clf.covariances_
    meens = clf.means_
    ws = clf.weights_
    fv = zeros(shape=shape(V1))
    if ndim(covs)>2:
        for k in range(0,shape(covs)[0]):
            U = squeeze(linalg.inv(covs[k,:,:]))
            v10 = meens[k,i]
            v20 = meens[k,j]
            U0 = array([[U[i,i],U[i,j]],[U[j,i],U[j,j]]])
            V = vstack((U[i,:],U[j,:]))    
            V = delete(V, (i,j), axis=1)

            W = delete(U, (i,j), axis=0)
            W = delete(W, (i,j), axis=1)
            Uoff = linalg.multi_dot([V, linalg.inv(W), V.T])
            Ut = U0-Uoff
            V1o = V1-v10
            V2o = V2-v20
            #Norm = (1.0/sqrt(2*pi))*sqrt(linalg.det(W))  
            Norm = 1.0
            fv += ws[k]*Norm*exp(-0.5*(V1o**2.0*Ut[0,0]+V2o**2.0*Ut[1,1]+2*V1o*V2o*Ut[1,0]))  
    else:
        v10 = meens[0,i]
        v20 = meens[0,j]
        Sig_inv = 1.0/covs
        V1o = V1-v10
        V2o = V2-v20
        Norm = sqrt(Sig_inv[0,j]*Sig_inv[0,i])/(2*pi)
        fv = Norm*exp(-0.5*(V1o**2.0*Sig_inv[0,i]+V2o**2.0*Sig_inv[0,j]))
    fv = log(fv)
    fv = fv-amax(fv)
    return fv


def FitStars(Cand,RemoveOutliers = False,z_th = 6.0):
    # Get data
    name = Cand.group_id.unique()[0]
    nstars = size(Cand,0)
    feh = Cand.feh # metallicity
    vx,vy,vz = Cand.GalRVel,Cand.GalTVel,Cand.GalzVel # velocities
    x,y,z = Cand.GalRecX,Cand.GalRecY,Cand.GalRecZ # positions
    #x,y,z = Cand.GalR,Cand.Galphi,Cand.Galz # positions


    # Remove outliers if needed
    if RemoveOutliers and (nstars>15):
        x_red,y_red,z_red,vx_red,vy_red,vz_red,feh_red = RemovePhaseSpaceOutliers(x,y,z,vx,vy,vz,feh,z_th=z_th)
        data = array([x_red,y_red,z_red,vx_red,vy_red,vz_red,feh_red]).T
        nstars = size(x_red)  
    else:
        data = array([x,y,z,vx,vy,vz,feh]).T

    
    # Set up three models
    clfa = mixture.GaussianMixture(n_components=1, covariance_type='diag')
    clfb = mixture.GaussianMixture(n_components=1, covariance_type='full')
    clfc = mixture.GaussianMixture(n_components=2, covariance_type='full')

    # Fit data to each
    clfa.fit(data)
    clfb.fit(data)
    clfc.fit(data)  
    return data,clfa,clfb,clfc

def ResampleLocalStars(clf,Psun,v_meens,v_covs):
     # Local stars
    dataf = clf.sample(5e6)
    probsf = clf.predict_proba(dataf[0][:,:])
    distf = sqrt((Sun[0]-dataf[0][:,0])**2.0+(Sun[1]-dataf[0][:,1])**2.0+(Sun[2]-dataf[0][:,2])**2.0)

    #xf = dataf[0][:,0]*cos(dataf[0][:,1])
    #yf = dataf[0][:,0]*sin(dataf[0][:,1])
    #zf = dataf[0][:,2]
    #distf = sqrt((Sun_cyl[0]-xf)**2.0+(Sun_cyl[1]-yf)**2.0+(Sun_cyl[2]-zf)**2.0)
    
    
    if size(Psun)>1:
        probs = Psun
    else:
        probs = array([Psun])
        
    i = 0
    v_meens1 = zeros(shape=shape(v_meens))
    v_covs1 = zeros(shape=shape(v_covs))
    for psun in probs:
        if psun<1.2:
            mask = (distf<1.0)*(argmax(probsf,axis=1)==i)
            vxf = dataf[0][mask,3]
            vyf = dataf[0][mask,4]
            vzf = dataf[0][mask,5]
            #hx,vb = histogram(vxf,range=[vmin,vmax],bins=nbins_1D,normed=True)
            #hy,_  = histogram(vyf,range=[vmin,vmax],bins=nbins_1D,normed=True)
            #hz,_  = histogram(vzf,range=[vmin,vmax],bins=nbins_1D,normed=True)
            #vc = (vb[0:-1]+vb[1:])/2.0
            #ax_x.plot(vc,hx,'y-',lw=3)
            #ax_y.plot(vc,hy,'y-',lw=3)
            #ax_z.plot(vc,hz,'y-',lw=3)
            #print(i,'vx = ',mean(vxf),' ± ',std(vxf))
            #print(i,'vy = ',mean(vyf),' ± ',std(vyf))
            #print(i,'vz = ',mean(vzf),' ± ',std(vzf))
            v_meens1[i,:] = array([mean(vxf),mean(vyf),mean(vzf)])
            v_covs1[i,:,:] = cov(vstack((vxf,vyf,vzf)))
            if sum(v_meens1[i,:])==nan or sum(v_covs1[i,:,:])==nan:
                v_meens1[i,:] = v_meens[i,:]
                v_covs1[i,:,:] = v_covs[i,:,:]
        else:
            v_meens1[i,:] = v_meens[i,:]
            v_covs1[i,:,:] = v_covs[i,:,:]
        i += 1
    return v_meens1,v_covs1
    
def CountWraps(data,clfb,clfc):
    nstars = size(data,0)
    bics = array([0.0,0.0,0.0])
    #bics[0] = clfa.bic(data)
    #bics[1] = clfb.bic(data)
    #bics[2] = clfc.bic(data)
    # check if groups overlap and bimodal is overfitting
    #if argmin(bics)==2:
    covs = clfc.covariances_
    meens = clfc.means_
    chck = 0
    for k in range(3,6):
        dsig = 2.5*sqrt(covs[0,k,k])+2.5*sqrt(covs[1,k,k])
        dv = abs(meens[0,k]-meens[1,k])
        if dv>dsig:
            chck += 1
            
    if chck==0:
        bics[1] = -10000.0
    else:
        bics[2] = -10000.0
                  
    if (argmin(bics)==0) or (argmin(bics)==1) or (nstars<10):
        covs = clfb.covariances_
        meens = clfb.means_
        fehs = array([meens[0,6],sqrt(covs[0,6,6])])
        pops = shape(data)[0]
        
        x_meens = meens[:,0:3]
        x_covs = covs[:,0:3,0:3]
        Psun = SunProb(x_meens,x_covs)
        v_meens,v_covs = ResampleLocalStars(clfb,Psun,meens[:,3:6],covs[:,3:6,3:6])
        
    else:
        covs = clfc.covariances_
        meens = clfc.means_
   
        vv1 = meens[0,3:6]
        vv2 = meens[1,3:6]
        r1 = sqrt((data[:,3]-vv1[0])**2.0+(data[:,4]-vv1[1])**2.0+(data[:,5]-vv1[2])**2.0)
        r2 = sqrt((data[:,3]-vv2[0])**2.0+(data[:,4]-vv2[1])**2.0+(data[:,5]-vv2[2])**2.0)
        pops_both = array([sum(r1<r2),sum(r2<r1)])
        if pops_both[0]<3.0:
            data_red = data[r2<r1,:]
            clf_red = mixture.GaussianMixture(n_components=1, covariance_type='full')
            clf_red.fit(data_red)
            covs = clf_red.covariances_
            meens = clf_red.means_
            pops = pops_both[1]
            fehs = array([meens[0,6],sqrt(covs[0,6,6])])
            x_meens = meens[:,0:3]
            x_covs = covs[:,0:3,0:3]
            Psun = SunProb(x_meens,x_covs)
            v_meens,v_covs = ResampleLocalStars(clfc,Psun,meens[:,3:6],covs[:,3:6,3:6])
        elif pops_both[1]<3.0:
            data_red = data[r1<r2,:]
            clf_red = mixture.GaussianMixture(n_components=1, covariance_type='full')
            clf_red.fit(data_red)
            covs = clf_red.covariances_
            meens = clf_red.means_
            pops = pops_both[0]
            fehs = array([meens[0,6],sqrt(covs[0,6,6])])
            x_meens = meens[:,0:3]
            x_covs = covs[:,0:3,0:3]
            Psun = SunProb(x_meens,x_covs)
            v_meens,v_covs = ResampleLocalStars(clfc,Psun,meens[:,3:6],covs[:,3:6,3:6])
        else:
            fehs = zeros(shape=(2,2))
            fehs[0,:] = array([meens[0,6],sqrt(covs[0,6,6])])
            fehs[1,:] = array([meens[1,6],sqrt(covs[1,6,6])])
            pops = pops_both  
            x_meens = meens[:,0:3]
            x_covs = covs[:,0:3,0:3]
            Psun = SunProb(x_meens,x_covs)
            v_meens,v_covs = ResampleLocalStars(clfc,Psun,meens[:,3:6],covs[:,3:6,3:6])
    if sum(v_meens)==nan or sum(v_covs)==nan:
        v_meens[i,:] = meens[:,3:6]
        v_covs[i,:,:] = covs[:,3:6,3:6]
    return x_meens,x_covs,v_meens,v_covs,fehs,pops,Psun


def VelocityTriangle(Cand,vmin=-595.0,vmax=595.0,nfine=500,nbins_1D = 50,\
                            levels=[-6.2,-2.3,0],\
                            tit_fontsize=30,\
                            z_th = 6.0,\
                            RemoveOutliers = False,\
                            cmap=cm.Greens,\
                            col_hist='mediumseagreen',\
                            colp = 'mediumseagreen',\
                            col_a = 'purple',\
                            col_b = 'tomato',\
                            col_c = 'mediumblue',\
                            point_size = 8,\
                            lblsize = 31,\
                            xlblsize = 35,\
                            def_alph = 0.2,\
                            SaveFigure = True,\
                            PlotFullSample=False):

    
    ######
    name = Cand.group_id.unique()[0]
    nstars = size(Cand,0)
    vx,vy,vz = Cand.GalRVel,Cand.GalTVel,Cand.GalzVel
    x,y,z = Cand.GalRecX,Cand.GalRecY,Cand.GalRecZ
    feh = Cand.feh
    x_red,y_red,z_red,vx_red,vy_red,vz_red,feh_red = RemovePhaseSpaceOutliers(x,y,z,vx,vy,vz,feh,z_th=6)


    # Fit stars
    data,clfa,clfb,clfc = FitStars(Cand,RemoveOutliers=RemoveOutliers,z_th=z_th)

    # Choose Model
    x_meens,x_covs,v_meens,v_covs,fehs,pops,Psun = CountWraps(data,clfb,clfc)
    
    vfine = linspace(vmin,vmax,nfine)
    V1,V2 = meshgrid(vfine,vfine)

    # Set plot rc params
    plt.rcParams['axes.linewidth'] = 2.5
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    fig, axarr = plt.subplots(3, 3,figsize=(14,14))
    gs = gridspec.GridSpec(3, 3)
    gs.update(wspace=0.0,hspace=0.0)
    ax_x = plt.subplot(gs[0,0])
    ax_y = plt.subplot(gs[1,1])
    ax_z = plt.subplot(gs[2,2])

    ax_yx = plt.subplot(gs[1,0])
    ax_zx = plt.subplot(gs[2,0])
    ax_zy = plt.subplot(gs[2,1])

    fig.delaxes(plt.subplot(gs[0,1]))
    fig.delaxes(plt.subplot(gs[0,2]))
    fig.delaxes(plt.subplot(gs[1,2]))

    # 1D plots
    plt.sca(ax_x)
    ax_x.hist(vx,range=[vmin,vmax],bins=nbins_1D,color=col_hist,linewidth=3,alpha=def_alph,density=True,stacked=True)
    plt.hist(vx,range=[vmin,vmax],bins=nbins_1D,color=col_hist,linewidth=3,histtype='step',density=True,stacked=True)
    plt.plot(vfine,fv_1D(vfine,clfb,3),'-',linewidth=3,color=col_b)
    plt.plot(vfine,fv_1D(vfine,clfc,3),'-',linewidth=3,color=col_c)
    plt.ylabel(r'$v_R$ [km s$^{-1}$]',fontsize=xlblsize)

    plt.sca(ax_y)
    ax_y.hist(vy,range=[vmin,vmax],bins=nbins_1D,color=col_hist,linewidth=3,alpha=def_alph,density=True,stacked=True)
    plt.hist(vy,range=[vmin,vmax],bins=nbins_1D,color=col_hist,linewidth=3,histtype='step',density=True,stacked=True)
    plt.plot(vfine,fv_1D(vfine,clfb,4),'-',linewidth=3,color=col_b)
    plt.plot(vfine,fv_1D(vfine,clfc,4),'-',linewidth=3,color=col_c)

    plt.sca(ax_z)
    ax_z.hist(vz,range=[vmin,vmax],bins=nbins_1D,color=col_hist,linewidth=3,alpha=def_alph,density=True,stacked=True)
    plt.hist(vz,range=[vmin,vmax],bins=nbins_1D,color=col_hist,linewidth=3,histtype='step',density=True,stacked=True)
    plt.plot(vfine,fv_1D(vfine,clfb,5),'-',linewidth=3,color=col_b)
    plt.plot(vfine,fv_1D(vfine,clfc,5),'-',linewidth=3,color=col_c)
    plt.xlabel(r'$v_z$ [km s$^{-1}$]',fontsize=xlblsize)


    # 2D plots
    plt.sca(ax_yx)
    ax_yx.plot(vx_red,vy_red,'o',markersize=point_size,markerfacecolor=colp,markeredgecolor=colp,label='Stars')
    ax_yx.plot(vx,vy,'o',markersize=point_size,markerfacecolor='none',markeredgecolor=colp,label='Outliers')
    ax_yx.contour(vfine,vfine,fv_2D(V1,V2,clfb,3,4),levels=levels,colors=col_b,linewidths=3,linestyles='solid')
    ax_yx.contour(vfine,vfine,fv_2D(V1,V2,clfc,3,4),levels=levels,colors=col_c,linewidths=3,linestyles='solid')
    plt.ylabel(r'$v_\phi$ [km s$^{-1}$]',fontsize=xlblsize)

    plt.sca(ax_zx)
    ax_zx.plot(vx_red,vz_red,'o',markersize=point_size,markerfacecolor=colp,markeredgecolor=colp)
    ax_zx.plot(vx,vz,'o',markersize=point_size,markerfacecolor='none',markeredgecolor=colp)
    ax_zx.contour(vfine,vfine,fv_2D(V1,V2,clfb,3,5),levels=levels,colors=col_b,linewidths=3,linestyles='solid')
    ax_zx.contour(vfine,vfine,fv_2D(V1,V2,clfc,3,5),levels=levels,colors=col_c,linewidths=3,linestyles='solid')
    plt.xlabel(r'$v_R$ [km s$^{-1}$]',fontsize=xlblsize)
    plt.ylabel(r'$v_z$ [km s$^{-1}$]',fontsize=xlblsize)

    plt.sca(ax_zy)
    ax_zy.plot(vy_red,vz_red,'o',markersize=point_size,markerfacecolor=colp,markeredgecolor=colp)
    ax_zy.plot(vy,vz,'o',markersize=point_size,markerfacecolor='none',markeredgecolor=colp)
    ax_zy.contour(vfine,vfine,fv_2D(V1,V2,clfb,4,5),levels=levels,colors=col_b,linewidths=3,linestyles='solid')
    ax_zy.contour(vfine,vfine,fv_2D(V1,V2,clfc,4,5),levels=levels,colors=col_c,linewidths=3,linestyles='solid')
    plt.xlabel(r'$v_\phi$ [km s$^{-1}$]',fontsize=xlblsize)

    # Tick style
    ax_x.tick_params(which='major',direction='in',width=2,length=10,right=True,top=True,pad=7,labelsize=24)
    ax_y.tick_params(which='major',direction='in',width=2,length=10,right=True,top=True,pad=7,labelsize=24)
    ax_z.tick_params(which='major',direction='in',width=2,length=10,right=True,top=True,pad=7,labelsize=24)
    ax_zx.tick_params(which='major',direction='in',width=2,length=10,right=True,top=True,pad=7,labelsize=24)
    ax_yx.tick_params(which='major',direction='in',width=2,length=10,right=True,top=True,pad=7,labelsize=24)
    ax_zy.tick_params(which='major',direction='in',width=2,length=10,right=True,top=True,pad=7,labelsize=24)

    # fix x-ticks
    vtx = array([-500,-250,0,250,500])
    ax_zx.set_xticks(vtx)
    ax_zy.set_xticks(vtx)
    ax_z.set_xticks(vtx)
    ax_zx.set_xticklabels(vtx,rotation=40)
    ax_zy.set_xticklabels(vtx,rotation=40)
    ax_z.set_xticklabels(vtx,rotation=40)
    
    # Limits and removing ticks
    ax_yx.set_xlim([vmin,vmax])
    ax_yx.set_ylim([vmin,vmax])
    ax_zx.set_xlim([vmin,vmax])
    ax_zx.set_ylim([vmin,vmax])
    ax_zy.set_xlim([vmin,vmax])
    ax_zy.set_ylim([vmin,vmax])
    ax_x.set_xlim([vmin,vmax])
    ax_y.set_xlim([vmin,vmax])
    ax_z.set_xlim([vmin,vmax])
    ax_x.set_yticks([])
    ax_y.set_yticks([])
    ax_z.set_yticks([])
    ax_x.set_yticklabels([])
    ax_x.set_xticklabels([])
    ax_y.set_yticklabels([])
    ax_y.set_xticklabels([])
    ax_z.set_yticklabels([])
    ax_yx.set_xticklabels([])
    ax_zy.set_yticklabels([])


    # Label for name and number of stars
    xlab = 0.66
    plt.gcf().text(xlab, 0.84, r'\bf {'+name+r'}', fontsize=60)
    plt.gcf().text(xlab,0.805,str(nstars)+' stars',fontsize=30)
                  
 
    label_b = '1 wrap'
    label_c = '2 wraps'
    if size(Psun)==1:
        ax_x.fill_between(vfine,fv_1D(vfine,clfb,3),facecolor=col_b,alpha=def_alph,zorder=-5)
        ax_y.fill_between(vfine,fv_1D(vfine,clfb,4),facecolor=col_b,alpha=def_alph,zorder=-5)
        ax_z.fill_between(vfine,fv_1D(vfine,clfb,5),facecolor=col_b,alpha=def_alph,zorder=-5)
        ax_yx.contourf(vfine,vfine,fv_2D(V1,V2,clfb,3,4),levels=levels,colors=col_b,alpha=def_alph,zorder=-5)
        ax_zx.contourf(vfine,vfine,fv_2D(V1,V2,clfb,3,5),levels=levels,colors=col_b,alpha=def_alph,zorder=-5)
        ax_zy.contourf(vfine,vfine,fv_2D(V1,V2,clfb,4,5),levels=levels,colors=col_b,alpha=def_alph,zorder=-5)

        
        
        plt.sca(ax_yx)
        ax_yx.fill_between(-10000*vfine/vfine,-1000*vfine/vfine,\
                           y2=-10000,lw=3,edgecolor=col_b,facecolor=col_alpha(col_b),label=label_b,zorder=-1)
        ax_yx.plot(10*vmin,-10*vmin,'-',lw=3,color=col_c,label=label_c,zorder=5)
        plt.gcf().text(xlab,0.77,r'{\bf Wraps = 1}',fontsize=30,color=col_b) 
    
        plt.gcf().text(xlab,0.72,r'$\bar{v}_R $ = '\
                       +'{:.1f}'.format(v_meens[0,0])\
                       +'$\pm$'+'{:.1f}'.format(sqrt(v_covs[0,0,0]))\
                       +' km s$^{-1}$',fontsize=25)           
        plt.gcf().text(xlab,0.69,r'$\bar{v}_\phi $ = '\
                       +'{:.1f}'.format(v_meens[0,1])\
                       +'$\pm$'+'{:.1f}'.format(sqrt(v_covs[0,1,1]))\
                       +' km s$^{-1}$',fontsize=25)            
        plt.gcf().text(xlab,0.66,r'$\bar{v}_z $ = '\
                       +'{:.1f}'.format(v_meens[0,2])\
                       +'$\pm$'+'{:.1f}'.format(sqrt(v_covs[0,2,2]))\
                       +' km s$^{-1}$',fontsize=25) 
        
        plt.gcf().text(xlab,0.63,r'[Fe/H] = '\
                       +'{:.1f}'.format(fehs[0])\
                       +'$\pm$'+'{:.1f}'.format(fehs[1]),fontsize=25) 
        
        plt.gcf().text(xlab,0.59,r'$P(\mathbf{x}_\odot)$ = '+'{:.1f}'.format(Psun)+r'$\sigma$',fontsize=25)

    else:
        ax_x.fill_between(vfine,fv_1D(vfine,clfc,3),facecolor=col_c,alpha=def_alph,zorder=-5)
        ax_y.fill_between(vfine,fv_1D(vfine,clfc,4),facecolor=col_c,alpha=def_alph,zorder=-5)
        ax_z.fill_between(vfine,fv_1D(vfine,clfc,5),facecolor=col_c,alpha=def_alph,zorder=-5)
        ax_yx.contourf(vfine,vfine,fv_2D(V1,V2,clfc,3,4),levels=levels,colors=col_c,alpha=def_alph,zorder=-5)
        ax_zx.contourf(vfine,vfine,fv_2D(V1,V2,clfc,3,5),levels=levels,colors=col_c,alpha=def_alph,zorder=-5)
        ax_zy.contourf(vfine,vfine,fv_2D(V1,V2,clfc,4,5),levels=levels,colors=col_c,alpha=def_alph,zorder=-5)

        plt.sca(ax_yx)
        
        plt.gcf().text(xlab,0.77,r'{\bf Wraps = 2}',fontsize=30,color=col_c) 
            
        ax_yx.plot(10*vmin,-10*vmin,'-',lw=3,color=col_b,label=label_b)
        ax_yx.fill_between(-10000*vfine/vfine,-1000*vfine/vfine,\
                           y2=-10000,lw=3,edgecolor=col_c,facecolor=col_alpha(col_c),label=label_c)
        plt.gcf().text(xlab,0.72,r'$\bar{v}_{R,1} $ = '\
                       +'{:.1f}'.format(v_meens[0,0])\
                       +'$\pm$'+'{:.1f}'.format(sqrt(v_covs[0,0,0]))\
                       +' km s$^{-1}$',fontsize=25)           
        plt.gcf().text(xlab,0.69,r'$\bar{v}_{\phi,1} $ = '\
                       +'{:.1f}'.format(v_meens[0,1])\
                       +'$\pm$'+'{:.1f}'.format(sqrt(v_covs[0,1,1]))\
                       +' km s$^{-1}$',fontsize=25)            
        plt.gcf().text(xlab,0.66,r'$\bar{v}_{z,1} $ = '\
                       +'{:.1f}'.format(v_meens[0,2])\
                       +'$\pm$'+'{:.1f}'.format(sqrt(v_covs[0,2,2]))\
                       +' km s$^{-1}$',fontsize=25)  
        plt.gcf().text(xlab,0.63,r'[Fe/H]$_1$ = '+'{:.1f}'.format(fehs[0,0])+'$\pm$'+'{:.1f}'.format(fehs[0,1]),fontsize=25) 
        plt.gcf().text(xlab,0.60,r'$P(\mathbf{x}_\odot)_1$ = '+'{:.1f}'.format(Psun[0])+r'$\sigma$',fontsize=25)

        
        plt.gcf().text(xlab,0.53,r'$\bar{v}_{R,2} $ = '\
                       +'{:.1f}'.format(v_meens[1,0])\
                       +'$\pm$'+'{:.1f}'.format(sqrt(v_covs[1,0,0]))\
                       +' km s$^{-1}$',fontsize=25)           
        plt.gcf().text(xlab,0.50,r'$\bar{v}_{\phi,2}$ = '\
                       +'{:.1f}'.format(v_meens[1,1])\
                       +'$\pm$'+'{:.1f}'.format(sqrt(v_covs[1,1,1]))\
                       +' km s$^{-1}$',fontsize=25)            
        plt.gcf().text(xlab,0.47,r'$\bar{v}_{z,2}$ = '\
                       +'{:.1f}'.format(v_meens[1,2])\
                       +'$\pm$'+'{:.1f}'.format(sqrt(v_covs[1,2,2]))\
                       +' km s$^{-1}$',fontsize=25) 
        plt.gcf().text(xlab,0.44,r'[Fe/H]$_2$ = '+'{:.1f}'.format(fehs[1,0])+'$\pm$'+'{:.1f}'.format(fehs[1,1]),fontsize=25) 
        plt.gcf().text(xlab,0.41,r'$P(\mathbf{x}_\odot)_2$ = '+'{:.1f}'.format(Psun[1])+r'$\sigma$',fontsize=25)
    

    if PlotFullSample:
        cmap = cm.gray_r
        cmap.set_under('white', 1.0)
        df_full = pandas.read_csv('../data/Gaia-SDSS.csv')
        vx0 = df_full.GalRVel.values
        vy0 = df_full.GalphiVel.values
        vz0 = df_full.GalzVel.values
        ax_zy.hexbin(vy0,vz0,extent=(vmin,vmax,vmin,vmax),gridsize=30,cmap=cmap,\
                     vmin=1.0,vmax=6000.0,linewidths=1.0,bins='log',zorder=-10,alpha=1.0)
        ax_yx.hexbin(vx0,vy0,extent=(vmin,vmax,vmin,vmax),gridsize=30,cmap=cmap,\
                     vmin=1.0,vmax=6000.0,linewidths=1.0,bins='log',zorder=-10,alpha=1.0)
        ax_zx.hexbin(vx0,vz0,extent=(vmin,vmax,vmin,vmax),gridsize=30,cmap=cmap,\
                     vmin=1.0,vmax=6000.0,linewidths=1.0,bins='log',zorder=-10,alpha=1.0)
        
    plt.legend(fontsize=lblsize-5,frameon=False,bbox_to_anchor=(1.05, 2.0), loc=2, borderaxespad=0.)

    if SaveFigure:
        fig.savefig('../plots/stars/Vtriangle_'+name+'.pdf',bbox_inches='tight') 
    return x_meens,x_covs,v_meens,v_covs,fehs,pops,Psun,fig



def XY_XZ(Cand,z_th=6.0,xmin = 0.0,xmax = 16.0,StarsColour='Purple',\
          BulgeColour = 'Crimson',DiskColour = 'Blue',\
          cmap = cm.Greens,Grid = True,Footprint=True,T_Myr = 100.0,OrbitsOn=True):

    # Set plot rc params
    plt.rcParams['axes.linewidth'] = 2.5
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    
    fig, axarr = plt.subplots(1, 2,figsize=(17,7))
    gs = gridspec.GridSpec(1, 2)
    gs.update(wspace=0.26)
    ax_xy = plt.subplot(gs[0])
    ax_xz = plt.subplot(gs[1])

    
    # Bulge and disk
    th = linspace(-pi,pi,500)
    xvals = linspace(0.0,xmax,500)
    zvals = linspace(-xmax/2.0,xmax/2.0,500)
    xx,zz = meshgrid(xvals,zvals)
    R = sqrt(xx**2.0+zz**2.0)
    rp = sqrt(R**2.0+(zz/0.5)**2.0)
    rho_bulge = 95.6/((1.0+(rp/0.075))**1.8)*exp(-(rp/2.1)**2.0)
    rho_bulge_xy = 95.6/((1.0+(R/0.075))**1.8)*exp(-(R/2.1)**2.0)

    rho_thind = 816.6/(2*0.3)*exp(-abs(zz)/0.3 - R/2.6)
    rho_thickd = 209.5/(2*0.9)*exp(-abs(zz)/0.9 - R/3.6)
    rhomin = 0.5
    ax_xz.contourf(xvals,zvals,log10(rho_thind),levels=arange(rhomin,3.0,0.5),cmap=cm.Blues,alpha=0.5,zorder=-1)
    ax_xz.contourf(xvals,zvals,log10(rho_thickd),levels=arange(rhomin,3.0,0.5),cmap=cm.GnBu,alpha=0.5,zorder=-1)
    ax_xz.contourf(xvals,zvals,log10(rho_bulge),levels=arange(-2,3,0.5),cmap=cm.Reds,alpha=0.9,zorder=-1)
    ax_xy.contourf(xvals,zvals,log10(rho_bulge_xy),levels=arange(-2,3,0.5),cmap=cm.Reds,alpha=0.9,zorder=-1)
    #ax_xz.contour(xvals,zvals,log10(rho_bulge),\
          #levels=arange(-2,3,0.5),colors=BulgeColour,alpha=0.4,zorder=-1,linestyles='solid')
    #ax_xy.contour(xvals,zvals,log10(rho_bulge_xy),\
          #levels=arange(-2,3,0.5),colors=BulgeColour,alpha=0.4,zorder=-1,linestyles='solid')


    name = Cand.group_id.unique()[0]
    nstars = size(Cand,0)
    x,y,z = Cand.GalRecX,Cand.GalRecY,Cand.GalRecZ
    U,V,W = Cand.GalU,Cand.GalV,Cand.GalW

    # reduced points
    ZSCORE = abs(zscore(z))+abs(zscore(x))+abs(zscore(y))\
            +abs(zscore(U))+abs(zscore(V))+abs(zscore(W))
    x_red = x[ZSCORE<z_th]
    y_red = y[ZSCORE<z_th]
    z_red = z[ZSCORE<z_th]
    U_red = U[ZSCORE<z_th]
    V_red = V[ZSCORE<z_th]
    W_red = W[ZSCORE<z_th]

    # Convex hull full
    points = zeros(shape=(size(x),2))
    points[:,0] = x
    points[:,1] = y
    hull = ConvexHull(points)
    x_edge = points[hull.vertices,0]
    y_edge = points[hull.vertices,1]
    ax_xy.fill(x_edge,y_edge,alpha=0.4,color='gray',zorder=0)

    points = zeros(shape=(size(x),2))
    points[:,0] = x
    points[:,1] = z
    hull = ConvexHull(points)
    x_edge = points[hull.vertices,0]
    z_edge = points[hull.vertices,1]
    ax_xz.fill(x_edge,z_edge,alpha=0.4,color='gray',zorder=0)

    # Convex hull reduced
    points = zeros(shape=(size(x_red),2))
    points[:,0] = x_red
    points[:,1] = y_red
    hull = ConvexHull(points)
    x_edge = points[hull.vertices,0]
    y_edge = points[hull.vertices,1]
    hull_smooth = chaikins_corner_cutting(x_edge,y_edge)
    ax_xy.fill(hull_smooth[:,0],hull_smooth[:,1],alpha=0.6,color=StarsColour,zorder=0)

    points[:,1] = z_red
    hull = ConvexHull(points)
    x_edge = points[hull.vertices,0]
    z_edge = points[hull.vertices,1]
    hull_smooth = chaikins_corner_cutting(x_edge,z_edge)
    ax_xz.fill(hull_smooth[:,0],hull_smooth[:,1],alpha=0.6,color=StarsColour,zorder=0)

    # Arrows
    ax_xy.quiver(x,y,U,V,z,alpha=0.5,cmap=cmap,scale=3000.0,zorder=1)
    ax_xy.quiver(x,y,U,V,edgecolor='k', facecolor='None', linewidth=.5,scale=3000.0,zorder=1)
    ax_xz.quiver(x,z,U,W,-y,alpha=0.5,cmap=cmap,scale=3000.0,zorder=1)
    ax_xz.quiver(x,z,U,W,edgecolor='k', facecolor='None', linewidth=.5,scale=3000.0,zorder=1)




    
    
    # Total moving group arrow
    ax_xy.quiver(mean(x),mean(y),mean(U),mean(V),\
                 color='none',alpha=1.0,scale=1000.0,linewidth=1.5,edgecolor='k',width=0.01)
    ax_xz.quiver(mean(x),mean(z),mean(U),mean(W),\
                 color='none',alpha=1.0,scale=1000.0,linewidth=1.5,edgecolor='k',width=0.01)
    ax_xy.quiver(mean(x),mean(y),mean(U),mean(V),\
                 color='gray',alpha=0.5,scale=1000.0,linewidth=1.5,edgecolor='k',width=0.01)
    ax_xz.quiver(mean(x),mean(z),mean(U),mean(W),\
                 color='gray',alpha=0.5,scale=1000.0,linewidth=1.5,edgecolor='k',width=0.01)

    
    ax_xy.quiver(mean(x_red),mean(y_red),mean(U_red),mean(V_red),\
                 color=StarsColour,scale=1000.0,linewidth=1.5,edgecolor='k',width=0.01)
    ax_xz.quiver(mean(x_red),mean(z_red),mean(U_red),mean(W_red),\
                 color=StarsColour,scale=1000.0,linewidth=1.5,edgecolor='k',width=0.01)

   

    # xy labels
    ax_xy.tick_params(which='major',direction='in',width=2,length=10,right=True,top=True,pad=10,labelsize=20)
    ax_xy.tick_params(which='minor',direction='in',width=1,length=7,right=True,top=True)
    ax_xy.set_xlabel(r"Galactic $X$ [kpc]",fontsize=27);
    ax_xy.set_ylabel(r"Galactic $Y$ [kpc]",fontsize=27);

    # xz labels
    ax_xz.tick_params(which='major',direction='in',width=2,length=10,right=True,top=True,pad=10,labelsize=20)
    ax_xz.tick_params(which='minor',direction='in',width=1,length=7,right=True,top=True)
    ax_xz.set_xlabel(r"Galactic $X$ [kpc]",fontsize=27);
    ax_xz.set_ylabel(r"Galactic $Z$ [kpc]",fontsize=27);

    plt.gcf().text(0.89,0.85,r'\bf {'+name+r'}', fontsize=40,horizontalalignment='right',verticalalignment='top')  
    plt.gcf().text(0.45,0.85,r'\bf {'+name+r'}', fontsize=40,horizontalalignment='right',verticalalignment='top')  

    if Grid:
        phi = linspace(-pi/2,pi/2,7)
        xvals = linspace(xmin,xmax,100)
        rvals = arange(xmin,xmax*1.5,2.0)
        nr = size(rvals)
        zvals = arange(-xmax,xmax,2.0)
        nz = size(zvals)
        for ii in range(0,7):
            ax_xy.plot(xvals,xvals*tan(phi[ii]),'-',color='gray',lw=0.5)
        for ii in range(0,nr):
            ax_xy.plot(rvals[ii]*cos(th),rvals[ii]*sin(th),'-',color='gray',lw=0.5)
            ax_xz.plot(rvals[ii]*cos(th),rvals[ii]*sin(th),'-',color='gray',lw=0.5)
        for ii in range(0,nz):
            ax_xz.plot([xmin,xmax],[zvals[ii],zvals[ii]],'-',color='gray',lw=0.5)
        ax_xy.set_yticks(arange(-xmax,xmax,2.0))
        ax_xz.set_yticks(arange(-xmax,xmax,2.0))

    ax_xy.set_xlim([xmin,xmax])
    ax_xy.set_ylim([-xmax/2.0,xmax/2.0])
    ax_xz.set_xlim([xmin,xmax])
    ax_xz.set_ylim([-xmax/2.0,xmax/2.0])
    
    if Footprint:
        footprint_XY = loadtxt('../data/GAIA-SDSS_footprint_XY.txt')
        ax_xy.plot(footprint_XY[:,0],footprint_XY[:,1],'-',color='gray',lw=1.0)
        footprint_XZ = loadtxt('../data/GAIA-SDSS_footprint_XZ.txt')
        ax_xz.plot(footprint_XZ[:,0],footprint_XZ[:,1],'-',color='gray',lw=1.0)

    # orbital units
    kpc = units.kpc
    kms = units.km/units.s
    deg = units.deg
    Gyr = units.Gyr
    ts = linspace(0.0,T_Myr*units.Myr,500)

    # Sun
    x1 = Sun[0]*cos(-pi/4)+0.2
    y1 = Sun[0]*sin(-pi/4)-0.1
    x2 = Sun[0]*cos(-pi/4+0.1)+0.2
    y2 = Sun[0]*sin(-pi/4+0.1)-0.1
    ax_xy.arrow(x1,y1,x2-x1,y2-y1,color='orangered',lw=3,length_includes_head=True,head_width=0.5)

    x1 = Sun[0]*cos(pi/4)+0.2
    y1 = Sun[0]*sin(pi/4)+0.2
    x2 = Sun[0]*cos(pi/4+0.1)+0.2
    y2 = Sun[0]*sin(pi/4+0.1)+0.2
    ax_xy.arrow(x1,y1,x2-x1,y2-y1,color='orangered',lw=3,length_includes_head=True,head_width=0.5)
    
    o_sun1 = Orbit(vxvv=[Sun[0]*kpc,0.0*kms,232.0*kms,0.0*kpc,0.0*kms,0.0*deg]).flip()
    o_sun1.integrate(ts,MWPotential2014)
    o_sun = Orbit(vxvv=[Sun[0]*kpc,0.0*kms,232.0*kms,0.0*kpc,0.0*kms,0.0*deg])
    o_sun.integrate(ts,MWPotential2014)
    
    ax_xy.plot(o_sun1.x(ts),o_sun1.y(ts),'--',lw=3,color='orangered')
    ax_xy.plot(o_sun.x(ts),o_sun.y(ts),'--',lw=3,color='orangered')
    ax_xy.plot(Sun[0],Sun[1],'*',markerfacecolor='yellow',markersize=25,markeredgecolor='red',markeredgewidth=2)
    ax_xz.plot(Sun[0],Sun[2],'*',markerfacecolor='yellow',markersize=25,markeredgecolor='red',markeredgewidth=2)

    if OrbitsOn==True:
        # Stellar orbits
        col_orb = 'ForestGreen'
        for i in range(0,nstars):
            R = Cand.GalR[i]
            vR = Cand.GalRVel[i]
            vT = Cand.GalTVel[i]
            z = Cand.Galz[i]
            vz = Cand.GalzVel[i]
            phi = Cand.Galphi[i]*180/pi

            o1 = Orbit(vxvv=[R*kpc,vR*kms,vT*kms,z*kpc,vz*kms,phi*deg]).flip()
            o1.integrate(ts,MWPotential2014)
            o2 = Orbit(vxvv=[R*kpc,vR*kms,vT*kms,z*kpc,vz*kms,phi*deg])
            o2.integrate(ts,MWPotential2014)
            ax_xy.plot(o1.x(ts),o1.y(ts),'-',alpha=0.6,color=col_orb,lw=0.4,zorder=-10)
            ax_xy.plot(o2.x(ts),o2.y(ts),'-',alpha=0.6,color=col_orb,lw=0.4,zorder=-10)

            ax_xz.plot(o1.x(ts),o1.z(ts),'-',alpha=0.6,color=col_orb,lw=0.4,zorder=-10)
            ax_xz.plot(o2.x(ts),o2.z(ts),'-',alpha=0.6,color=col_orb,lw=0.4,zorder=-10)

        
        
        
    #sigr = std(Cand.GalRVel)
    #sigphi = std(Cand.GalTVel)
    #sigz = std(Cand.GalzVel)
    #beta = 1.0-(sigz**2.0+sigphi**2.0)/(2*sigr**2.0)
    #plt.gcf().text(0.79, 0.16, r'$\beta$ = '+r'{:.2f}'.format(beta), fontsize=30)
    fig.savefig('../plots/stars/XYZ_'+name+'.pdf',bbox_inches='tight')
    return fig




def Orbits(Cand,xlim=16.0,ylim=16.0,zlim=16.0,T_Myr=10.0):

    # Set plot rc params
    plt.rcParams['axes.linewidth'] = 2.5
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    
    fig = plt.figure(figsize=(16,15))
    ax = fig.gca(projection='3d')

    name = Cand.group_id.unique()[0]
    nstars = size(Cand,0)


    # disk
    def rhoR(x,y,z):
        R = sqrt(x**2 + y**2)
        rp = sqrt(R**2.0+(z/0.5)**2.0)
        return 95.6*1000/((1.0+(rp/0.075))**1.8)*exp(-(rp/2.1)**2.0)

    def rhoz1(x,y,z):
        R = sqrt(x**2 + y**2+z**2.0)
        return 816.6/(2*0.3)*exp(-abs(z)/0.3 - R/2.6)

    def rhoz2(x,y,z):
        R = sqrt(x**2 + y**2)
        return 209.5/(2*0.9)*exp(-abs(z)/0.9 - R/3.6)

    def rho_full(x,y,z):
        return rhoR(x,y,z)+rhoz1(x,y,z)+rhoz2(x,y,z)

    ni = 50
    xmin = -xlim
    xmax = xlim
    ymin = -ylim
    ymax = ylim
    zmin = 0.0
    zmax = 3.0
    X, Y, Z = meshgrid(linspace(xmin,xmax,ni),linspace(ymin,ymax,ni),linspace(zmin,zmax,ni))
    verts, faces, _, _ = measure.marching_cubes_lewiner(rho_full(X,Y,Z), rho_full(0.0,0.0,zmax),spacing=(1.0, 1.0, 1.0))
    xverts = verts[:, 0]*(xmax-xmin)/(ni-1) + xmin
    yverts = verts[:, 1]*(ymax-ymin)/(ni-1) + ymin
    zverts = verts[:, 2]*(zmax-zmin)/(ni-1) + zmin
    ax.plot_trisurf(xverts, yverts, faces, zverts,cmap=cm.coolwarm, lw=1,alpha=0.3,zorder=-10)


    # orbits
    kpc = units.kpc
    kms = units.km/units.s
    deg = units.deg
    Gyr = units.Gyr

    Flip = True

    ts = linspace(0.0,T_Myr*units.Myr,500)

    for i in range(0,nstars):
        R = Cand.GalR[i]
        vR = Cand.GalRVel[i]
        vT = Cand.GalTVel[i]
        z = Cand.Galz[i]
        vz = Cand.GalzVel[i]
        phi = Cand.Galphi[i]*180/pi
        # -t
        o = Orbit(vxvv=[R*kpc,vR*kms,vT*kms,z*kpc,vz*kms,phi*deg]).flip()
        o.integrate(ts,MWPotential2014)
        ax.plot(o.x(ts),o.y(ts),o.z(ts),'b-',color='ForestGreen')

        # +t
        o = Orbit(vxvv=[R*kpc,vR*kms,vT*kms,z*kpc,vz*kms,phi*deg])
        o.integrate(ts,MWPotential2014)
        ax.plot(o.x(ts),o.y(ts),o.z(ts),'-',color='ForestGreen')
        ax.scatter(o.x(),o.y(),o.z(),marker='o',s=100)



    # Sun -t
    o_sun1 = Orbit(vxvv=[Sun[0]*kpc,0.0*kms,232.0*kms,0.0*kpc,0.0*kms,0.0*deg]).flip()
    o_sun1.integrate(ts,MWPotential2014)
    ax.plot(o_sun1.x(ts),o_sun1.y(ts),o_sun1.z(ts),'k-',lw=3)
    # +t
    o_sun = Orbit(vxvv=[Sun[0]*kpc,0.0*kms,232.0*kms,0.0*kpc,0.0*kms,0.0*deg])
    o_sun.integrate(ts,MWPotential2014)
    ax.plot(o_sun.x(ts),o_sun.y(ts),o_sun.z(ts),'k-',lw=3)

    ax.scatter(o_sun.x(),o_sun.y(),o_sun.z(),s=2000,marker='*',color='orange',edgecolor='firebrick',lw=3,zorder=10)



    # Omega-cen
    o_cen = Orbit.from_name('Omega Cen').flip()
    o_cen.integrate(ts,MWPotential2014)
    ax.plot(o_cen.x(ts),o_cen.y(ts),o_cen.z(ts),'r-',lw=3)

    o_cen = Orbit.from_name('Omega Cen')
    o_cen.integrate(ts,MWPotential2014)
    ax.plot(o_cen.x(ts),o_cen.y(ts),o_cen.z(ts),'r-',lw=3)

    ax.scatter(o_cen.x(),o_cen.y(),o_cen.z(),s=500,marker='$\omega$',color='k')


    # Galactic center
    ax.scatter(0, 0, 0,marker='o',s=1000,color='r')

    ax.set_xlim3d([-xlim,xlim])
    ax.set_ylim3d([-ylim,ylim])
    ax.set_zlim3d([-zlim,zlim])
    ax.set_xlabel('Galactic $X$ [kpc]',fontsize=30,labelpad=30)
    ax.set_ylabel('Galactic $Y$ [kpc]',fontsize=30,labelpad=30)
    ax.set_zlabel('Galactic $Z$ [kpc]',fontsize=30,labelpad=30)
    ax.tick_params(which='major',direction='in',width=2,length=10,right=True,top=True,pad=10,labelsize=26)
    ax.tick_params(which='minor',direction='in',width=1,length=7,right=True,top=True)
    plt.gcf().text(0.6, 0.8, r'$\Delta T = [-$'+str(ts[-1])+',+'+str(ts[-1])+']',fontsize=30)
    plt.gcf().text(0.25, 0.78, r'\bf {'+name+r'}', fontsize=60)

    fig.savefig('../plots/stars/Orbits_'+name+'.pdf',bbox_inches='tight')

    return fig


def StreamOrbit(Cand,nt=100,T_Myr=10.0,Moving=False):
    Cand = Cand.reset_index()

    nstars = size(Cand,0)

    # orbits
    kpc = units.kpc
    kms = units.km/units.s
    deg = units.deg
    Gyr = units.Gyr

    ts = linspace(0.0,T_Myr*units.Myr,nt/2)
    t_tot = append(ts,-ts)
    rsun = zeros(shape=(nstars,nt))

    if Moving:
        osun1 = Orbit(vxvv=[Sun[0]*kpc,0.0*kms,232.0*kms,0.0*kpc,0.0*kms,0.0*deg]).flip()
        osun1.integrate(ts,MWPotential2014)
        osun2 = Orbit(vxvv=[Sun[0]*kpc,0.0*kms,232.0*kms,0.0*kpc,0.0*kms,0.0*deg])
        osun2.integrate(ts,MWPotential2014)
        osun1x,osun1y,osun1z = osun1.x(ts),osun1.y(ts),osun1.z(ts)
        osun2x,osun2y,osun2z = osun2.x(ts),osun2.y(ts),osun2.z(ts)
    else:
        osun1x,osun1y,osun1z = Sun[0],Sun[1],Sun[2]
        osun2x,osun2y,osun2z = Sun[0],Sun[1],Sun[2]
        
    for i in range(0,nstars):
        R = Cand.GalR[i]
        vR = Cand.GalRVel[i]
        vT = Cand.GalTVel[i]
        z = Cand.Galz[i]
        vz = Cand.GalzVel[i]
        phi = Cand.Galphi[i]*180/pi
        # -t
        o1 = Orbit(vxvv=[R*kpc,vR*kms,vT*kms,z*kpc,vz*kms,phi*deg]).flip()
        o1.integrate(ts,MWPotential2014)

        # +t
        o2 = Orbit(vxvv=[R*kpc,vR*kms,vT*kms,z*kpc,vz*kms,phi*deg])
        o2.integrate(ts,MWPotential2014)

  
        rsun[i,0:nt/2] = flipud(sqrt((o1.x(ts)-osun1x)**2.0+(o1.y(ts)-osun1y)**2.0+(o1.z(ts)-osun1z)**2.0))
        rsun[i,nt/2:] = (sqrt((o2.x(ts)-osun2x)**2.0+(o2.y(ts)-osun2y)**2.0+(o2.z(ts)-osun2z)**2.0))

    sig_sun = sqrt(2)*erfinv(amin(sum(rsun>1.0,0))/(1.0*nstars))
    rsun_sorted = sort(rsun,0)
    rsun1 = rsun_sorted[int((1-0.95)*nstars),:]
    rsun2 = rsun_sorted[int((1-0.68)*nstars),:]
    rsun3 = rsun_sorted[int(0.68*nstars),:]
    rsun4 = rsun_sorted[int(0.95*nstars),:]
    orb_env = vstack([rsun1,rsun2,rsun3,rsun4])
    t = linspace(-T_Myr,T_Myr,nt)
    return orb_env,rsun,sig_sun,t


def MollweideMap1(ax,TH,PH,fv0,cmin,cmax,nlevels,cmap,tfs,PlotCygnus=False,gridlinecolor='k',GalacticPlane=False):
    plt.rcParams['axes.linewidth'] = 3
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif',size=15)

    
    ax.contourf(rad2deg(PH), rad2deg(TH),fv0,nlevels,transform=ccrs.PlateCarree(),\
                cmap=cmap,vmin=cmin,vmax=cmax,linestyles='none',antialiased=True)

    gl = ax.gridlines(color=gridlinecolor,linewidth=1.5, linestyle='--',alpha=0.5)
    gl.ylocator = mticker.FixedLocator([-90,-60, -30, 0, 30, 60,90])
    ax.outline_patch.set_linewidth(3)
   

    tx = array([r'$-60^\circ$',r'$-30^\circ$',r'$0^\circ$',r'$+30^\circ$',r'$+60^\circ$']) 
    xtx = array([0.17,0.05,-0.01,0.05,0.18])
    ytx = array([0.08,0.26,0.49,0.72,0.9])
    
    for i in range(0,size(xtx)):
        plt.text(xtx[i],ytx[i],tx[i],transform=ax.transAxes,horizontalalignment='right',verticalalignment='center',fontsize=tfs)


    if PlotCygnus==True:
        ax.plot(-cyg[0:4,0],cyg[0:4,1],'-',color='crimson',transform=ccrs.PlateCarree())
        ax.plot(-cyg[4:,0],cyg[4:,1],'-',color='crimson',transform=ccrs.PlateCarree())
        ax.plot(-cyg[:,0],cyg[:,1],'.',color='k',ms=5,transform=ccrs.PlateCarree())

    if GalacticPlane==True:
        ax.plot([-181,181],[0,0],'-',color=gridlinecolor,lw=1.5,transform=ccrs.PlateCarree())
        ax.text(125,4,'Galactic',color=gridlinecolor,transform=ccrs.PlateCarree(),fontsize=int(tfs*0.8))
        ax.text(135,-10,'plane',color=gridlinecolor,transform=ccrs.PlateCarree(),fontsize=int(tfs*0.8))
    return

