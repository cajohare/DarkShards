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

Sun = array([8.122,0.0,0.005])

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


def VelocityTriangle(Cand,vmin=-495.0,vmax=495.0,nfine=500,nbins_1D = 50,\
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
                            lblsize = 28,\
                            def_alph = 0.1):

    
    ######
    name = Cand.group_id.unique()[0]
    nstars = size(Cand,0)
    feh = Cand.feh
    vx,vy,vz = Cand.GalRVel,Cand.GalTVel,Cand.GalzVel
    x,y,z = Cand.GalRecX,Cand.GalRecY,Cand.GalRecZ
    x_red,y_red,z_red,vx_red,vy_red,vz_red = RemovePhaseSpaceOutliers(x,y,z,vx,vy,vz,z_th=z_th)

    if RemoveOutliers:
        data = array([vx_red,vy_red,vz_red]).T
        nstars = size(vx_red)
    else:
        data = array([vx,vy,vz,feh]).T
        nstars = size(vx)
    clfa = mixture.GaussianMixture(n_components=1, covariance_type='diag')
    clfb = mixture.GaussianMixture(n_components=1, covariance_type='full')
    clfc = mixture.GaussianMixture(n_components=2, covariance_type='full')

    clfa.fit(data)
    clfb.fit(data)
    clfc.fit(data)

    vfine = linspace(vmin,vmax,nfine)
    V1,V2 = meshgrid(vfine,vfine)

    def fv_1D(clf,i):
        covs = clf.covariances_
        meens = clf.means_
        fv = zeros(shape=nfine)
        if ndim(covs)>2:
            for k in range(0,shape(covs)[0]):
                sig0_sq = covs[k,i,i]
                v0 = meens[k,i]
                Norm = (1.0/sqrt(2*pi*sig0_sq))
                fv += Norm*exp(-(vfine-v0)**2.0/(2*sig0_sq))
        else:
            sig0_sq = covs[0,i]
            v0 = meens[0,i]
            Norm = (1.0/sqrt(2*pi*sig0_sq))
            fv = Norm*exp(-(vfine-v0)**2.0/(2*sig0_sq))
        fv /= trapz(fv,vfine)
        return fv


    def fv_2D(clf,i,j):
        covs = clf.covariances_
        meens = clf.means_
        fv = zeros(shape=(nfine,nfine))
        if ndim(covs)>2:
            for k in range(0,shape(covs)[0]):
                v10 = meens[k,i]
                v20 = meens[k,j]
                Sig_inv = linalg.inv(covs[k,:,:])
                V1o = V1-v10
                V2o = V2-v20
                Norm = sqrt(Sig_inv[j,j]*Sig_inv[j,j])/(2*pi)
                fv += Norm*exp(-0.5*(V1o**2.0*Sig_inv[i,i]+V2o**2.0*Sig_inv[j,j]+2*V1o*V2o*Sig_inv[j,i]))
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
    plt.plot(vfine,fv_1D(clfa,0),'-',linewidth=3,color=col_a)
    plt.plot(vfine,fv_1D(clfb,0),'-',linewidth=3,color=col_b)
    plt.plot(vfine,fv_1D(clfc,0),'-',linewidth=3,color=col_c)
    #plt.title(r'$\langle v_r \rangle= $ '+str(int(vx0))+r' km s$^{-1}$',fontsize=tit_fontsize)
    plt.ylabel(r'$v_r$ [km s$^{-1}$]',fontsize=lblsize)

    plt.sca(ax_y)
    ax_y.hist(vy,range=[vmin,vmax],bins=nbins_1D,color=col_hist,linewidth=3,alpha=def_alph,normed=1)
    plt.hist(vy,range=[vmin,vmax],bins=nbins_1D,color=col_hist,linewidth=3,histtype='step',normed=1)
    plt.plot(vfine,fv_1D(clfa,1),'-',linewidth=3,color=col_a)
    plt.plot(vfine,fv_1D(clfb,1),'-',linewidth=3,color=col_b)
    plt.plot(vfine,fv_1D(clfc,1),'-',linewidth=3,color=col_c)
    #plt.title(r'$\langle v_\phi \rangle = $ '+str(int(vy0))+r' km s$^{-1}$',fontsize=tit_fontsize)

    plt.sca(ax_z)
    ax_z.hist(vz,range=[vmin,vmax],bins=nbins_1D,color=col_hist,linewidth=3,alpha=def_alph,normed=1)
    plt.hist(vz,range=[vmin,vmax],bins=nbins_1D,color=col_hist,linewidth=3,histtype='step',normed=1)
    plt.plot(vfine,fv_1D(clfa,2),'-',linewidth=3,color=col_a)
    plt.plot(vfine,fv_1D(clfb,2),'-',linewidth=3,color=col_b)
    plt.plot(vfine,fv_1D(clfc,2),'-',linewidth=3,color=col_c)
    #plt.title(r'$\langle v_z \rangle= $ '+str(int(vz0))+r' km s$^{-1}$',fontsize=tit_fontsize)
    plt.xlabel(r'$v_z$ [km s$^{-1}$]',fontsize=lblsize)


    # 2D plots
    plt.sca(ax_yx)
    ax_yx.plot(vx_red,vy_red,'o',markersize=point_size,markerfacecolor=colp,markeredgecolor=colp,label='Stars')
    ax_yx.plot(vx,vy,'o',markersize=point_size+3,markerfacecolor='none',markeredgecolor=colp,label='Outliers')
    ax_yx.contour(vfine,vfine,fv_2D(clfa,0,1),levels=levels,colors=col_a,linewidths=3,linestyles='solid')
    ax_yx.contour(vfine,vfine,fv_2D(clfb,0,1),levels=levels,colors=col_b,linewidths=3,linestyles='solid')
    ax_yx.contour(vfine,vfine,fv_2D(clfc,0,1),levels=levels,colors=col_c,linewidths=3,linestyles='solid')
    plt.ylabel(r'$v_\phi$ [km s$^{-1}$]',fontsize=lblsize)

    plt.sca(ax_zx)
    ax_zx.plot(vx_red,vz_red,'o',markersize=point_size,markerfacecolor=colp,markeredgecolor=colp)
    ax_zx.plot(vx,vz,'o',markersize=point_size+3,markerfacecolor='none',markeredgecolor=colp)
    ax_zx.contour(vfine,vfine,fv_2D(clfa,0,2),levels=levels,colors=col_a,linewidths=3,linestyles='solid')
    ax_zx.contour(vfine,vfine,fv_2D(clfb,0,2),levels=levels,colors=col_b,linewidths=3,linestyles='solid')
    ax_zx.contour(vfine,vfine,fv_2D(clfc,0,2),levels=levels,colors=col_c,linewidths=3,linestyles='solid')
    plt.xlabel(r'$v_r$ [km s$^{-1}$]',fontsize=lblsize)
    plt.ylabel(r'$v_z$ [km s$^{-1}$]',fontsize=lblsize)

    plt.sca(ax_zy)
    ax_zy.plot(vy_red,vz_red,'o',markersize=point_size,markerfacecolor=colp,markeredgecolor=colp)
    ax_zy.plot(vy,vz,'o',markersize=point_size+3,markerfacecolor='none',markeredgecolor=colp)
    ax_zy.contour(vfine,vfine,fv_2D(clfa,1,2),levels=levels,colors=col_a,linewidths=3,linestyles='solid')
    ax_zy.contour(vfine,vfine,fv_2D(clfb,1,2),levels=levels,colors=col_b,linewidths=3,linestyles='solid')
    ax_zy.contour(vfine,vfine,fv_2D(clfc,1,2),levels=levels,colors=col_c,linewidths=3,linestyles='solid')
    plt.xlabel(r'$v_\phi$ [km s$^{-1}$]',fontsize=lblsize)

    ax_x.tick_params(which='major',direction='in',width=2,length=10,right=True,top=True,pad=7,labelsize=23)
    ax_y.tick_params(which='major',direction='in',width=2,length=10,right=True,top=True,pad=7,labelsize=23)
    ax_z.tick_params(which='major',direction='in',width=2,length=10,right=True,top=True,pad=7,labelsize=23)
    ax_zx.tick_params(which='major',direction='in',width=2,length=10,right=True,top=True,pad=7,labelsize=23)
    ax_yx.tick_params(which='major',direction='in',width=2,length=10,right=True,top=True,pad=7,labelsize=23)
    ax_zy.tick_params(which='major',direction='in',width=2,length=10,right=True,top=True,pad=7,labelsize=23)


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
    Psun = 1-chi2.cdf(dL,3)

    plt.gcf().text(xlab,0.77,r'$P(\mathbf{x}_\odot)$ = '+'{:.1f}'.format(sqrt(2)*erfinv(Psun[0]))+r'$\sigma$',fontsize=30)

    #r'$\langle v_r \rangle $ = '\+'{:.1f}'.format(meens[0,0])+'$\pm$'+'{:.1f}'.format(sqrt(covs[0,0,0]))+' km s$^{-1}$',fontsize=25)   

    # "LEGEND"
    bics = array([0.0,0.0,0.0])
    data = array([x,y,z,vx,vy,vz,feh]).T
    clfa_full = mixture.GaussianMixture(n_components=1, covariance_type='diag')
    clfb_full = mixture.GaussianMixture(n_components=1, covariance_type='full')
    clfc_full = mixture.GaussianMixture(n_components=2, covariance_type='full')
    clfa_full.fit(data)
    clfb_full.fit(data)
    clfc_full.fit(data)
    bics[0] = clfa_full.bic(data)
    bics[1] = clfb_full.bic(data)
    bics[2] = clfc_full.bic(data)

    def col_alpha(col,alpha=def_alph):
        rgb = colors.colorConverter.to_rgb(col)
        bg_rgb = [1,1,1]
        return [alpha * c1 + (1 - alpha) * c2
                for (c1, c2) in zip(rgb, bg_rgb)]

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
                  

    label_a = '1 mode (diag $\sigma$)'
    label_b = '1 mode (full $\sigma$)'
    label_c = '2 modes (full $\sigma$)'
    if (argmin(bics)==0) or (argmin(bics)==1):
        ax_x.fill_between(vfine,fv_1D(clfb,0),facecolor=col_b,alpha=def_alph,zorder=-5)
        ax_y.fill_between(vfine,fv_1D(clfb,1),facecolor=col_b,alpha=def_alph,zorder=-5)
        ax_z.fill_between(vfine,fv_1D(clfb,2),facecolor=col_b,alpha=def_alph,zorder=-5)
        ax_yx.contourf(vfine,vfine,fv_2D(clfb,0,1),levels=levels,colors=col_b,alpha=def_alph,zorder=-5)
        ax_zx.contourf(vfine,vfine,fv_2D(clfb,0,2),levels=levels,colors=col_b,alpha=def_alph,zorder=-5)
        ax_zy.contourf(vfine,vfine,fv_2D(clfb,1,2),levels=levels,colors=col_b,alpha=def_alph,zorder=-5)

        plt.sca(ax_yx)
        ax_yx.plot(10*vmin,-10*vmin,'-',lw=3,color=col_a,label=label_a,zorder=-10)
        ax_yx.fill_between(-10000*vfine/vfine,-1000*vfine/vfine,y2=-10000,lw=3,edgecolor=col_b,facecolor=col_alpha(col_b),label=label_b,zorder=-1)
        ax_yx.plot(10*vmin,-10*vmin,'-',lw=3,color=col_c,label=label_c,zorder=5)
        plt.gcf().text(xlab,0.74,r'{\bf Groups = 1}',fontsize=30,color=col_b) 

        covs = clfc.covariances_
        meens = clfc.means_
        plt.gcf().text(xlab,0.705,r'$\langle v_r \rangle $ = '\
                       +'{:.1f}'.format(meens[0,0])\
                       +'$\pm$'+'{:.1f}'.format(sqrt(covs[0,0,0]))\
                       +' km s$^{-1}$',fontsize=25)           
        plt.gcf().text(xlab,0.675,r'$\langle v_\phi \rangle $ = '\
                       +'{:.1f}'.format(meens[0,1])\
                       +'$\pm$'+'{:.1f}'.format(sqrt(covs[0,1,1]))\
                       +' km s$^{-1}$',fontsize=25)            
        plt.gcf().text(xlab,0.645,r'$\langle v_z \rangle $ = '\
                       +'{:.1f}'.format(meens[0,2])\
                       +'$\pm$'+'{:.1f}'.format(sqrt(covs[0,2,2]))\
                       +' km s$^{-1}$',fontsize=25) 
    else:
        ax_x.fill_between(vfine,fv_1D(clfc,0),facecolor=col_c,alpha=def_alph,zorder=-5)
        ax_y.fill_between(vfine,fv_1D(clfc,1),facecolor=col_c,alpha=def_alph,zorder=-5)
        ax_z.fill_between(vfine,fv_1D(clfc,2),facecolor=col_c,alpha=def_alph,zorder=-5)
        ax_yx.contourf(vfine,vfine,fv_2D(clfc,0,1),levels=levels,colors=col_c,alpha=def_alph,zorder=-5)
        ax_zx.contourf(vfine,vfine,fv_2D(clfc,0,2),levels=levels,colors=col_c,alpha=def_alph,zorder=-5)
        ax_zy.contourf(vfine,vfine,fv_2D(clfc,1,2),levels=levels,colors=col_c,alpha=def_alph,zorder=-5)

        plt.sca(ax_yx)
        ax_yx.plot(10*vmin,-10*vmin,'-',lw=3,color=col_a,label=label_a)
        ax_yx.plot(10*vmin,-10*vmin,'-',lw=3,color=col_b,label=label_b)
        ax_yx.fill_between(-10000*vfine/vfine,-1000*vfine/vfine,y2=-10000,lw=3,edgecolor=col_c,facecolor=col_alpha(col_c),label=label_c)
        covs = clfc.covariances_
        meens = clfc.means_
        plt.gcf().text(xlab,0.705,r'$\langle v_r \rangle $ = '\
                       +'{:.1f}'.format(meens[0,0])\
                       +'$\pm$'+'{:.1f}'.format(sqrt(covs[0,0,0]))\
                       +' km s$^{-1}$',fontsize=23)           
        plt.gcf().text(xlab,0.675,r'$\langle v_\phi \rangle $ = '\
                       +'{:.1f}'.format(meens[0,1])\
                       +'$\pm$'+'{:.1f}'.format(sqrt(covs[0,1,1]))\
                       +' km s$^{-1}$',fontsize=23)            
        plt.gcf().text(xlab,0.645,r'$\langle v_z \rangle $ = '\
                       +'{:.1f}'.format(meens[0,2])\
                       +'$\pm$'+'{:.1f}'.format(sqrt(covs[0,2,2]))\
                       +' km s$^{-1}$',fontsize=23)  

        plt.gcf().text(xlab,0.59,r'$\langle v_r \rangle $ = '\
                       +'{:.1f}'.format(meens[1,0])\
                       +'$\pm$'+'{:.1f}'.format(sqrt(covs[1,0,0]))\
                       +' km s$^{-1}$',fontsize=23)           
        plt.gcf().text(xlab,0.56,r'$\langle v_\phi \rangle $ = '\
                       +'{:.1f}'.format(meens[1,1])\
                       +'$\pm$'+'{:.1f}'.format(sqrt(covs[1,1,1]))\
                       +' km s$^{-1}$',fontsize=23)            
        plt.gcf().text(xlab,0.53,r'$\langle v_z \rangle $ = '\
                       +'{:.1f}'.format(meens[1,2])\
                       +'$\pm$'+'{:.1f}'.format(sqrt(covs[1,2,2]))\
                       +' km s$^{-1}$',fontsize=23) 

        plt.gcf().text(xlab,0.74,r'{\bf Groups = 2}',fontsize=30,color=col_c) 



    plt.legend(fontsize=lblsize-3,frameon=False,bbox_to_anchor=(1.05, 2.0), loc=2, borderaxespad=0.)

    # SunOverlap = in_hull(Sun,transpose(array([x,y,z])))
    # if SunOverlap:
    #     plt.gcf().text(0.66,0.675,r'Sun in full hull?' ,fontsize=30,color='ForestGreen')
    # else:
    #     plt.gcf().text(0.66,0.675,r'Sun in full hull?',fontsize=30,color='Crimson')

    # SunOverlap = in_hull(Sun,transpose(array([x_red,y_red,z_red])))
    # if SunOverlap:
    #     plt.gcf().text(0.66,0.64,r'Sun in reduced hull?' ,fontsize=30,color='ForestGreen')
    # else:
    #     plt.gcf().text(0.66,0.64,r'Sun in reduced hull?',fontsize=30,color='Crimson')

    fig.savefig('../plots/stars/Vtriangle_'+name+'.pdf',bbox_inches='tight') 
    return fig





def XY_XZ(Cand,z_th=6.0,xmin = 0.0,xmax = 16.0,StarsColour='Purple',\
          BulgeColour = 'Crimson',DiskColour = 'Blue',\
          cmap = cm.Greens,Grid = True,Footprint=True):

    # Set plot rc params
    plt.rcParams['axes.linewidth'] = 2.5
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    
    fig, axarr = plt.subplots(1, 2,figsize=(16,7))
    gs = gridspec.GridSpec(1, 2)
    gs.update(wspace=0.23)
    ax_xy = plt.subplot(gs[0])
    ax_xz = plt.subplot(gs[1])

    
    # Bulge and disk
    th = linspace(-pi,pi,100)
    xvals = linspace(0.0,xmax,100)
    zvals = linspace(-xmax/2.0,xmax/2.0,100)
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
    ax_xy.fill(x_edge,y_edge,alpha=0.4,color=StarsColour,zorder=0)

    points = zeros(shape=(size(x),2))
    points[:,0] = x
    points[:,1] = z
    hull = ConvexHull(points)
    x_edge = points[hull.vertices,0]
    z_edge = points[hull.vertices,1]
    ax_xz.fill(x_edge,z_edge,alpha=0.4,color=StarsColour,zorder=0)

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



    # The sun
    ax_xy.plot(Sun[0]*cos(th),Sun[0]*sin(th),'--',linewidth=3,color='orangered')
    ax_xy.plot(Sun[0],Sun[1],'*',markerfacecolor='yellow',markersize=25,markeredgecolor='red',markeredgewidth=2)
    ax_xz.plot(Sun[0],Sun[2],'*',markerfacecolor='yellow',markersize=25,markeredgecolor='red',markeredgewidth=2)
    x1 = Sun[0]*cos(-pi/4)
    y1 = Sun[0]*sin(-pi/4)
    x2 = Sun[0]*cos(-pi/4+0.1)
    y2 = Sun[0]*sin(-pi/4+0.1)
    ax_xy.arrow(x1,y1,x2-x1,y2-y1,color='orangered',lw=3,length_includes_head=True,head_width=0.5)

    x1 = Sun[0]*cos(pi/4)
    y1 = Sun[0]*sin(pi/4)
    x2 = Sun[0]*cos(pi/4+0.1)
    y2 = Sun[0]*sin(pi/4+0.1)
    ax_xy.arrow(x1,y1,x2-x1,y2-y1,color='orangered',lw=3,length_includes_head=True,head_width=0.5)

    
    
    # Total moving group arrow
    ax_xy.quiver(mean(x_red),mean(y_red),mean(U_red),mean(V_red),\
                 color=StarsColour,scale=1000.0,linewidth=1.5,edgecolor='k',width=0.01)
    ax_xz.quiver(mean(x_red),mean(z_red),mean(U_red),mean(W_red),\
                 color=StarsColour,scale=1000.0,linewidth=1.5,edgecolor='k',width=0.01)

    ax_xy.quiver(mean(x),mean(y),mean(U),mean(V),\
                 color=StarsColour,alpha=0.75,scale=1000.0,linewidth=1.5,edgecolor='k',width=0.01)
    ax_xz.quiver(mean(x),mean(z),mean(U),mean(W),\
                 color=StarsColour,alpha=0.75,scale=1000.0,linewidth=1.5,edgecolor='k',width=0.01)


    # xy labels
    ax_xy.tick_params(which='major',direction='in',width=2,length=10,right=True,top=True,pad=10,labelsize=15)
    ax_xy.tick_params(which='minor',direction='in',width=1,length=7,right=True,top=True)
    ax_xy.set_xlabel(r"Galactic $X$ [kpc]",fontsize=27);
    ax_xy.set_ylabel(r"Galactic $Y$ [kpc]",fontsize=27);

    # xz labels
    ax_xz.tick_params(which='major',direction='in',width=2,length=10,right=True,top=True,pad=10,labelsize=15)
    ax_xz.tick_params(which='minor',direction='in',width=1,length=7,right=True,top=True)
    ax_xz.set_xlabel(r"Galactic $X$ [kpc]",fontsize=27);
    ax_xz.set_ylabel(r"Galactic $Z$ [kpc]",fontsize=27);

    plt.gcf().text(0.46,0.86,r'\bf {'+name+r'}', fontsize=40,horizontalalignment='right',verticalalignment='top')  

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
        footprint_XY = loadtxt('../GAIA-SDSS_footprint_XY.txt')
        ax_xy.plot(footprint_XY[:,0],footprint_XY[:,1],'-',color='gray',lw=1.0)
        footprint_XZ = loadtxt('../GAIA-SDSS_footprint_XZ.txt')
        ax_xz.plot(footprint_XZ[:,0],footprint_XZ[:,1],'-',color='gray',lw=1.0)


    sigr = std(Cand.GalRVel)
    sigphi = std(Cand.GalTVel)
    sigz = std(Cand.GalzVel)
    beta = 1.0-(sigz**2.0+sigphi**2.0)/(2*sigr**2.0)
    plt.gcf().text(0.79, 0.16, r'$\beta$ = '+r'{:.2f}'.format(beta), fontsize=30)
    fig.savefig('../plots/stars/XYZ_'+name+'.pdf',bbox_inches='tight')
    return fig


from galpy.orbit import Orbit
from galpy.potential import MWPotential2014
from mpl_toolkits.mplot3d import Axes3D
from astropy import units
from skimage import measure

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
        phi = Cand.Galphi[i]
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

