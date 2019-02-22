from numpy import *
from numpy.random import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.basemap import Basemap
import matplotlib.gridspec as gridspec
from matplotlib import colors
import pandas
from scipy.spatial import Delaunay
from scipy.spatial import ConvexHull
from scipy.stats import zscore,chi2
from sklearn import mixture
from scipy.special import erfinv


# Galpy
from galpy.orbit import Orbit
from galpy.potential import MWPotential2014
from mpl_toolkits.mplot3d import Axes3D
from astropy import units
from skimage import measure

Sun = array([8.122,0.0,0.005])


def MySquarePlot(xlab,ylab,lw=2.5,lfs=45,tfs=25,size_x=13,size_y=12,Grid=False):
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

def MyDoublePlot(xlab1,ylab1,xlab2,ylab2,wspace=0.25,lw=2.5,lfs=45,tfs=25,size_x=20,size_y=11,Grid=False):
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


def RemovePhaseSpaceOutliers(x,y,z,U,V,W,z_th=6.0):
    # reduced points
    nstars = size(x)
    if nstars>5:
        ZSCORE = abs(zscore(z))+abs(zscore(x))+abs(zscore(y))\
                +abs(zscore(U))+abs(zscore(V))+abs(zscore(W))
    else:
        ZSCORE = (z_th-1)*ones(shape=nstars)    
    x_red = x[ZSCORE<z_th]
    y_red = y[ZSCORE<z_th]
    z_red = z[ZSCORE<z_th]
    U_red = U[ZSCORE<z_th]
    V_red = V[ZSCORE<z_th]
    W_red = W[ZSCORE<z_th]
    return x_red,y_red,z_red,U_red,V_red,W_red


def FitStars(Cand,RemoveOutliers = False,z_th = 6.0):

    # Get data
    name = Cand.group_id.unique()[0]
    nstars = size(Cand,0)
    feh = Cand.feh # metallicity
    vx,vy,vz = Cand.GalRVel,Cand.GalTVel,Cand.GalzVel # velocities
    x,y,z = Cand.GalRecX,Cand.GalRecY,Cand.GalRecZ # positions

    # Remove outliers if needed
    if RemoveOutliers:
        x_red,y_red,z_red,vx_red,vy_red,vz_red = RemovePhaseSpaceOutliers(x,y,z,vx,vy,vz,z_th=z_th)
        data = array([x_red,y_red,z_red,vx_red,vy_red,vz_red,feh]).T
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
    
    # Calculate Bayesian information criterion
    bics = array([0.0,0.0,0.0])
    bics[0] = clfa.bic(data)
    bics[1] = clfb.bic(data)
    bics[2] = clfc.bic(data)

    # Second check if bimodal distribution is overfitting
    if argmin(bics)==2:
        covs = clfc.covariances_
        meens = clfc.means_
        chck = 0
        for k in range(3,6):
            dsig = 3*sqrt(covs[0,k,k])+3*sqrt(covs[1,k,k])
            dv = abs(meens[0,k]-meens[1,k])
            if dv>dsig:
                chck += 1
            
        if chck==0:
            bics[1] = -10000.0
                  
    if (argmin(bics)==0) or (argmin(bics)==1) or (nstars<10):
        covs = clfb.covariances_
        meens = clfb.means_
        fehs = array([meens[0,6],sqrt(covs[0,6,6])])
        pops = shape(data)[0]
        
    else:
        covs = clfc.covariances_
        meens = clfc.means_
        fehs = zeros(shape=(2,2))
        fehs[0,:] = array([meens[0,6],sqrt(covs[0,6,6])])
        fehs[1,:] = array([meens[1,6],sqrt(covs[1,6,6])])
        
        vv1 = meens[0,3:6]
        vv2 = meens[1,3:6]
        r1 = sqrt((data[:,3]-vv1[0])**2.0+(data[:,4]-vv1[1])**2.0+(data[:,5]-vv1[2])**2.0)
        r2 = sqrt((data[:,3]-vv2[0])**2.0+(data[:,4]-vv2[1])**2.0+(data[:,5]-vv2[2])**2.0)
        pops = array([sum(r1<r2),sum(r2<r1)])
    
    x_meens = meens[:,0:3]
    v_meens = meens[:,3:6]
    x_covs = covs[:,0:3,0:3]
    v_covs = covs[:,3:6,3:6]

        
    # Sun overlap (just use positional data)
    clf_xyz = mixture.GaussianMixture(n_components=1, covariance_type='full')
    clf_xyz.fit(array([x,y,z]).T)
    lsun = clf_xyz.score_samples(Sun.reshape(-1,1).T)
    xyz_meens = clf_xyz.means_
    lmax = clf_xyz.score_samples(xyz_meens)
    dL = -2*(lsun-lmax)
    Psun = sqrt(2)*erfinv(chi2.cdf(dL,3))

    return x_meens,x_covs,v_meens,v_covs,fehs,pops,Psun





# see http://www-biba.inrialpes.fr/Jaynes/cappe1.pdf

########

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

def VelocityTriangle(Cand,vmin=-595.0,vmax=595.0,nfine=500,nbins_1D = 50,\
                            levels=[-6.2,-2.3,0],\
                            tit_fontsize=30,\
                            z_th = 6.0,\
                            RemoveOutliers = False,\
                            cmap=cm.Greens,\
                            col_hist='ForestGreen',\
                            colp = 'ForestGreen',\
                            col_a = 'tomato',\
                            col_b = 'purple',\
                            col_c = 'dodgerblue',\
                            point_size = 8,\
                            lblsize = 31,\
                            xlblsize = 35,\
                            def_alph = 0.2):

    
    ######
    name = Cand.group_id.unique()[0]
    nstars = size(Cand,0)
    feh = Cand.feh
    vx,vy,vz = Cand.GalRVel,Cand.GalTVel,Cand.GalzVel
    x,y,z = Cand.GalRecX,Cand.GalRecY,Cand.GalRecZ
    x_red,y_red,z_red,vx_red,vy_red,vz_red = RemovePhaseSpaceOutliers(x,y,z,vx,vy,vz,z_th=z_th)

    # Remove outliers if needed
    if RemoveOutliers:
        x_red,y_red,z_red,vx_red,vy_red,vz_red = RemovePhaseSpaceOutliers(x,y,z,vx,vy,vz,z_th=z_th)
        data = array([vx_red,vy_red,vz_red,x_red,y_red,z_red,feh]).T
    else:
        data = array([vx,vy,vz,x,y,z,feh]).T
        
    clfa = mixture.GaussianMixture(n_components=1, covariance_type='diag')
    clfb = mixture.GaussianMixture(n_components=1, covariance_type='full')
    clfc = mixture.GaussianMixture(n_components=2, covariance_type='full')

    clfa.fit(data)
    clfb.fit(data)
    clfc.fit(data)
   
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
    ax_x.hist(vx,range=[vmin,vmax],bins=nbins_1D,color=col_hist,linewidth=3,alpha=def_alph,normed=1)
    plt.hist(vx,range=[vmin,vmax],bins=nbins_1D,color=col_hist,linewidth=3,histtype='step',normed=1)
    plt.plot(vfine,fv_1D(vfine,clfa,0),'-',linewidth=5,color=col_a)
    plt.plot(vfine,fv_1D(vfine,clfb,0),'-',linewidth=3,color=col_b)
    plt.plot(vfine,fv_1D(vfine,clfc,0),'-',linewidth=3,color=col_c)
    plt.ylabel(r'$v_r$ [km s$^{-1}$]',fontsize=xlblsize)

    plt.sca(ax_y)
    ax_y.hist(vy,range=[vmin,vmax],bins=nbins_1D,color=col_hist,linewidth=3,alpha=def_alph,normed=1)
    plt.hist(vy,range=[vmin,vmax],bins=nbins_1D,color=col_hist,linewidth=3,histtype='step',normed=1)
    plt.plot(vfine,fv_1D(vfine,clfa,1),'-',linewidth=5,color=col_a)
    plt.plot(vfine,fv_1D(vfine,clfb,1),'-',linewidth=3,color=col_b)
    plt.plot(vfine,fv_1D(vfine,clfc,1),'-',linewidth=3,color=col_c)

    plt.sca(ax_z)
    ax_z.hist(vz,range=[vmin,vmax],bins=nbins_1D,color=col_hist,linewidth=3,alpha=def_alph,normed=1)
    plt.hist(vz,range=[vmin,vmax],bins=nbins_1D,color=col_hist,linewidth=3,histtype='step',normed=1)
    plt.plot(vfine,fv_1D(vfine,clfa,2),'-',linewidth=5,color=col_a)
    plt.plot(vfine,fv_1D(vfine,clfb,2),'-',linewidth=3,color=col_b)
    plt.plot(vfine,fv_1D(vfine,clfc,2),'-',linewidth=3,color=col_c)
    plt.xlabel(r'$v_z$ [km s$^{-1}$]',fontsize=xlblsize)


    # 2D plots
    plt.sca(ax_yx)
    ax_yx.plot(vx_red,vy_red,'o',markersize=point_size,markerfacecolor=colp,markeredgecolor=colp,label='Stars')
    ax_yx.plot(vx,vy,'o',markersize=point_size+3,markerfacecolor='none',markeredgecolor=colp,label='Outliers')
    ax_yx.contour(vfine,vfine,fv_2D(V1,V2,clfa,0,1),levels=levels,colors=col_a,linewidths=3,linestyles='solid')
    ax_yx.contour(vfine,vfine,fv_2D(V1,V2,clfb,0,1),levels=levels,colors=col_b,linewidths=3,linestyles='solid')
    ax_yx.contour(vfine,vfine,fv_2D(V1,V2,clfc,0,1),levels=levels,colors=col_c,linewidths=3,linestyles='solid')
    plt.ylabel(r'$v_\phi$ [km s$^{-1}$]',fontsize=xlblsize)

    plt.sca(ax_zx)
    ax_zx.plot(vx_red,vz_red,'o',markersize=point_size,markerfacecolor=colp,markeredgecolor=colp)
    ax_zx.plot(vx,vz,'o',markersize=point_size+3,markerfacecolor='none',markeredgecolor=colp)
    ax_zx.contour(vfine,vfine,fv_2D(V1,V2,clfa,0,2),levels=levels,colors=col_a,linewidths=3,linestyles='solid')
    ax_zx.contour(vfine,vfine,fv_2D(V1,V2,clfb,0,2),levels=levels,colors=col_b,linewidths=3,linestyles='solid')
    ax_zx.contour(vfine,vfine,fv_2D(V1,V2,clfc,0,2),levels=levels,colors=col_c,linewidths=3,linestyles='solid')
    plt.xlabel(r'$v_r$ [km s$^{-1}$]',fontsize=xlblsize)
    plt.ylabel(r'$v_z$ [km s$^{-1}$]',fontsize=xlblsize)

    plt.sca(ax_zy)
    ax_zy.plot(vy_red,vz_red,'o',markersize=point_size,markerfacecolor=colp,markeredgecolor=colp)
    ax_zy.plot(vy,vz,'o',markersize=point_size+3,markerfacecolor='none',markeredgecolor=colp)
    ax_zy.contour(vfine,vfine,fv_2D(V1,V2,clfa,1,2),levels=levels,colors=col_a,linewidths=3,linestyles='solid')
    ax_zy.contour(vfine,vfine,fv_2D(V1,V2,clfb,1,2),levels=levels,colors=col_b,linewidths=3,linestyles='solid')
    ax_zy.contour(vfine,vfine,fv_2D(V1,V2,clfc,1,2),levels=levels,colors=col_c,linewidths=3,linestyles='solid')
    plt.xlabel(r'$v_\phi$ [km s$^{-1}$]',fontsize=xlblsize)

    ax_x.tick_params(which='major',direction='in',width=2,length=10,right=True,top=True,pad=7,labelsize=24)
    ax_y.tick_params(which='major',direction='in',width=2,length=10,right=True,top=True,pad=7,labelsize=24)
    ax_z.tick_params(which='major',direction='in',width=2,length=10,right=True,top=True,pad=7,labelsize=24)
    ax_zx.tick_params(which='major',direction='in',width=2,length=10,right=True,top=True,pad=7,labelsize=24)
    ax_yx.tick_params(which='major',direction='in',width=2,length=10,right=True,top=True,pad=7,labelsize=24)
    ax_zy.tick_params(which='major',direction='in',width=2,length=10,right=True,top=True,pad=7,labelsize=24)


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

    xlab = 0.66
    plt.gcf().text(xlab, 0.84, r'\bf {'+name+r'}', fontsize=60)
    plt.gcf().text(xlab,0.805,str(nstars)+' stars',fontsize=30)
    #plt.gcf().text(x_lab,0.8,r'$\langle$[Fe/H]$\rangle$ = '+r'{:.2f}'.format(mean(feh)),fontsize=30)  

    # Sun overlap
    clf_xyz = mixture.GaussianMixture(n_components=1, covariance_type='full')
    clf_xyz.fit(array([x,y,z]).T)
    lsun = clf_xyz.score_samples(Sun.reshape(-1,1).T)
    xyz_meens = clf_xyz.means_
    lmax = clf_xyz.score_samples(xyz_meens)
    dL = -2*(lsun-lmax)
    Psun = sqrt(2)*erfinv(chi2.cdf(dL,3))
    plt.gcf().text(xlab,0.77,r'$P(\mathbf{x}_\odot)$ = '+'{:.1f}'.format(Psun[0])+r'$\sigma$',fontsize=30)

    # Choose model
    bics = array([0.0,0.0,0.0])
    bics[0] = clfa.bic(data)
    bics[1] = clfb.bic(data)
    bics[2] = clfc.bic(data)

    # check if groups overlap and bimodal is overfitting
    if argmin(bics)==2:
        covs = clfc.covariances_
        meens = clfc.means_
        chck = 0
        for k in range(0,3):
            dsig = 3*sqrt(covs[0,k,k])+3*sqrt(covs[1,k,k])
            dv = abs(meens[0,k]-meens[1,k])
            if dv>dsig:
                chck += 1
            
        if chck==0:
            bics[1] = -10000.0
                  

    label_a = '1 mode (diag $\Sigma$)'
    label_b = '1 mode (full $\Sigma$)'
    label_c = '2 modes (full $\Sigma$)'
    if (argmin(bics)==0) or (argmin(bics)==1) or (nstars<10):
        ax_x.fill_between(vfine,fv_1D(vfine,clfb,0),facecolor=col_b,alpha=def_alph,zorder=-5)
        ax_y.fill_between(vfine,fv_1D(vfine,clfb,1),facecolor=col_b,alpha=def_alph,zorder=-5)
        ax_z.fill_between(vfine,fv_1D(vfine,clfb,2),facecolor=col_b,alpha=def_alph,zorder=-5)
        ax_yx.contourf(vfine,vfine,fv_2D(V1,V2,clfb,0,1),levels=levels,colors=col_b,alpha=def_alph,zorder=-5)
        ax_zx.contourf(vfine,vfine,fv_2D(V1,V2,clfb,0,2),levels=levels,colors=col_b,alpha=def_alph,zorder=-5)
        ax_zy.contourf(vfine,vfine,fv_2D(V1,V2,clfb,1,2),levels=levels,colors=col_b,alpha=def_alph,zorder=-5)

        plt.sca(ax_yx)
        ax_yx.plot(10*vmin,-10*vmin,'-',lw=3,color=col_a,label=label_a,zorder=-10)
        ax_yx.fill_between(-10000*vfine/vfine,-1000*vfine/vfine,\
                           y2=-10000,lw=3,edgecolor=col_b,facecolor=col_alpha(col_b),label=label_b,zorder=-1)
        ax_yx.plot(10*vmin,-10*vmin,'-',lw=3,color=col_c,label=label_c,zorder=5)
        plt.gcf().text(xlab,0.74,r'{\bf Groups = 1}',fontsize=30,color=col_b) 

        covs = clfc.covariances_
        meens = clfc.means_
        plt.gcf().text(xlab,0.705,r'$\bar{v}_r $ = '\
                       +'{:.1f}'.format(meens[0,0])\
                       +'$\pm$'+'{:.1f}'.format(sqrt(covs[0,0,0]))\
                       +' km s$^{-1}$',fontsize=25)           
        plt.gcf().text(xlab,0.675,r'$\bar{v}_\phi $ = '\
                       +'{:.1f}'.format(meens[0,1])\
                       +'$\pm$'+'{:.1f}'.format(sqrt(covs[0,1,1]))\
                       +' km s$^{-1}$',fontsize=25)            
        plt.gcf().text(xlab,0.645,r'$\bar{v}_z $ = '\
                       +'{:.1f}'.format(meens[0,2])\
                       +'$\pm$'+'{:.1f}'.format(sqrt(covs[0,2,2]))\
                       +' km s$^{-1}$',fontsize=25) 
        
        plt.gcf().text(xlab,0.615,r'[Fe/H] = '\
                       +'{:.1f}'.format(meens[0,-1])\
                       +'$\pm$'+'{:.1f}'.format(sqrt(covs[0,-1,-1])),fontsize=25) 
    else:
        ax_x.fill_between(vfine,fv_1D(vfine,clfc,0),facecolor=col_c,alpha=def_alph,zorder=-5)
        ax_y.fill_between(vfine,fv_1D(vfine,clfc,1),facecolor=col_c,alpha=def_alph,zorder=-5)
        ax_z.fill_between(vfine,fv_1D(vfine,clfc,2),facecolor=col_c,alpha=def_alph,zorder=-5)
        ax_yx.contourf(vfine,vfine,fv_2D(V1,V2,clfc,0,1),levels=levels,colors=col_c,alpha=def_alph,zorder=-5)
        ax_zx.contourf(vfine,vfine,fv_2D(V1,V2,clfc,0,2),levels=levels,colors=col_c,alpha=def_alph,zorder=-5)
        ax_zy.contourf(vfine,vfine,fv_2D(V1,V2,clfc,1,2),levels=levels,colors=col_c,alpha=def_alph,zorder=-5)

        plt.sca(ax_yx)
        ax_yx.plot(10*vmin,-10*vmin,'-',lw=3,color=col_a,label=label_a)
        ax_yx.plot(10*vmin,-10*vmin,'-',lw=3,color=col_b,label=label_b)
        ax_yx.fill_between(-10000*vfine/vfine,-1000*vfine/vfine,\
                           y2=-10000,lw=3,edgecolor=col_c,facecolor=col_alpha(col_c),label=label_c)
        covs = clfc.covariances_
        meens = clfc.means_
        plt.gcf().text(xlab,0.705,r'$\bar{v}_r $ = '\
                       +'{:.1f}'.format(meens[0,0])\
                       +'$\pm$'+'{:.1f}'.format(sqrt(covs[0,0,0]))\
                       +' km s$^{-1}$',fontsize=25)           
        plt.gcf().text(xlab,0.675,r'$\bar{v}_\phi $ = '\
                       +'{:.1f}'.format(meens[0,1])\
                       +'$\pm$'+'{:.1f}'.format(sqrt(covs[0,1,1]))\
                       +' km s$^{-1}$',fontsize=25)            
        plt.gcf().text(xlab,0.645,r'$\bar{v}_z $ = '\
                       +'{:.1f}'.format(meens[0,2])\
                       +'$\pm$'+'{:.1f}'.format(sqrt(covs[0,2,2]))\
                       +' km s$^{-1}$',fontsize=25)  

        plt.gcf().text(xlab,0.59,r'$\bar{v}_r $ = '\
                       +'{:.1f}'.format(meens[1,0])\
                       +'$\pm$'+'{:.1f}'.format(sqrt(covs[1,0,0]))\
                       +' km s$^{-1}$',fontsize=25)           
        plt.gcf().text(xlab,0.56,r'$\bar{v}_\phi $ = '\
                       +'{:.1f}'.format(meens[1,1])\
                       +'$\pm$'+'{:.1f}'.format(sqrt(covs[1,1,1]))\
                       +' km s$^{-1}$',fontsize=25)            
        plt.gcf().text(xlab,0.53,r'$\bar{v}_z $ = '\
                       +'{:.1f}'.format(meens[1,2])\
                       +'$\pm$'+'{:.1f}'.format(sqrt(covs[1,2,2]))\
                       +' km s$^{-1}$',fontsize=25) 
        plt.gcf().text(xlab,0.50,r'[Fe/H] = '\
                       +'{:.1f}'.format(meens[0,-1])\
                       +'$\pm$'+'{:.1f}'.format(sqrt(covs[0,-1,-1])),fontsize=25) 

        plt.gcf().text(xlab,0.74,r'{\bf Groups = 2}',fontsize=30,color=col_c) 
       
    plt.legend(fontsize=lblsize-5,frameon=False,bbox_to_anchor=(1.05, 2.0), loc=2, borderaxespad=0.)

    fig.savefig('../plots/stars/Vtriangle_'+name+'.pdf',bbox_inches='tight') 
    return fig





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

