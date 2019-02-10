from numpy import *
from numpy.random import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.basemap import Basemap
import matplotlib.gridspec as gridspec
import pandas
from scipy.spatial import Delaunay

def in_hull(p, hull):
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p)>=0

def VelocityTriangle(Cand,\
                     vmin=-495.0,vmax=495.0,nbins=30,\
                     cmap=cm.Greens,col1='ForestGreen',\
                     levels=[0.2],\
                     tit_fontsize=30):
    
    name = Cand.group_id[0]
    nstars = size(Cand,0)
    feh = Cand.feh
    vx,vy,vz = Cand.GalRVel,Cand.GalTVel,Cand.GalzVel
    x,y,z = Cand.GalRecX,Cand.GalRecY,Cand.GalRecZ
    
    nstars = size(vx)
    vx0 = mean(vx)
    vy0 = mean(vy)
    vz0 = mean(vz)
    sigx = std(vx)
    sigy = std(vy)
    sigz = std(vz)

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

    vv = linspace(vmin,vmax,nbins)
    vfine = linspace(vmin,vmax,1000)
    V1,V2 = meshgrid(vv,vv)

    def fv_1D(v1):
        v10 = mean(v1)
        sig1 = std(v1)
        fv1 = exp(-(vfine-v10)**2.0/(2*sig1**2.0))
        fv1 /= trapz(fv1,vfine)
        return fv1

    def fv_2D(v1,v2):
        v10 = mean(v1)
        v20 = mean(v2)
        sig1 = std(v1)
        sig2 = std(v2)
        return exp(-(V1-v10)**2.0/(2*sig1**2.0) - (V2-v20)**2.0/(2*sig2**2.0))

    def fv_2D_tilt(v1,v2):
        v10 = mean(v1)
        v20 = mean(v2)
        Sig_inv= linalg.inv(cov(v1,v2))
        V1o = V1-v10
        V2o = V2-v20
        return exp(-0.5*(V1o**2.0*Sig_inv[0,0]+V2o**2.0*Sig_inv[1,1]+2*V1o*V2o*Sig_inv[1,0]))


    # 1D plots
    plt.sca(ax_x)
    ax_x.hist(vx,range=[vmin,vmax],bins=nbins,color=col1,linewidth=3,alpha=0.3,normed=1)
    plt.hist(vx,range=[vmin,vmax],bins=nbins,color=col1,linewidth=3,histtype='step',normed=1)
    plt.plot(vfine,fv_1D(vx),'k-',linewidth=3)
    plt.xlim([vmin,vmax])
    plt.title(r'$\langle v_r \rangle= $ '+str(int(vx0))+r' km s$^{-1}$',fontsize=tit_fontsize)
    plt.ylabel(r'$v_r$ [km s$^{-1}$]',fontsize=25)

    plt.sca(ax_y)
    ax_y.hist(vy,range=[vmin,vmax],bins=nbins,color=col1,linewidth=3,alpha=0.3,normed=1)
    plt.hist(vy,range=[vmin,vmax],bins=nbins,color=col1,linewidth=3,histtype='step',normed=1)
    plt.plot(vfine,fv_1D(vy),'k-',linewidth=3)
    plt.xlim([vmin,vmax])
    plt.title(r'$\langle v_\phi \rangle = $ '+str(int(vy0))+r' km s$^{-1}$',fontsize=tit_fontsize)


    plt.sca(ax_z)
    ax_z.hist(vz,range=[vmin,vmax],bins=nbins,color=col1,linewidth=3,alpha=0.3,normed=1)
    plt.hist(vz,range=[vmin,vmax],bins=nbins,color=col1,linewidth=3,histtype='step',normed=1)
    plt.plot(vfine,fv_1D(vz),'k-',linewidth=3)
    plt.xlim([vmin,vmax])
    plt.title(r'$\langle v_z \rangle= $ '+str(int(vz0))+r' km s$^{-1}$',fontsize=tit_fontsize)
    plt.xlabel(r'$v_z$ [km s$^{-1}$]',fontsize=25)


    # 2D plots
    plt.sca(ax_yx)
    ax_yx.hist2d(vx,vy,range=[[vmin, vmax], [vmin, vmax]],bins=nbins,cmap=cmap)
    ax_yx.contour(vv,vv,fv_2D(vx,vy),levels=levels,colors='k',linewidths=3)
    ax_yx.contour(vv,vv,fv_2D_tilt(vx,vy),levels=levels,colors='Crimson',linewidths=3)
    plt.ylabel(r'$v_\phi$ [km s$^{-1}$]',fontsize=25)

    plt.sca(ax_zx)
    ax_zx.hist2d(vx,vz,range=[[vmin, vmax], [vmin, vmax]],bins=nbins,cmap=cmap)
    ax_zx.contour(vv,vv,fv_2D(vx,vz),levels=levels,colors='k',linewidths=3)
    ax_zx.contour(vv,vv,fv_2D_tilt(vx,vz),levels=levels,colors='Crimson',linewidths=3)
    plt.xlabel(r'$v_r$ [km s$^{-1}$]',fontsize=25)
    plt.ylabel(r'$v_z$ [km s$^{-1}$]',fontsize=25)

    plt.sca(ax_zy)
    ax_zy.hist2d(vy,vz,range=[[vmin, vmax], [vmin, vmax]],bins=nbins,cmap=cmap)
    ax_zy.contour(vv,vv,fv_2D(vy,vz),levels=levels,colors='k',linewidths=3)
    ax_zy.contour(vv,vv,fv_2D_tilt(vy,vz),levels=levels,colors='Crimson',linewidths=3)
    plt.xlabel(r'$v_\phi$ [km s$^{-1}$]',fontsize=25)

    ax_x.tick_params(which='major',direction='in',width=2,length=10,right=True,top=True,pad=7,labelsize=23)
    ax_y.tick_params(which='major',direction='in',width=2,length=10,right=True,top=True,pad=7,labelsize=23)
    ax_z.tick_params(which='major',direction='in',width=2,length=10,right=True,top=True,pad=7,labelsize=23)
    ax_zx.tick_params(which='major',direction='in',width=2,length=10,right=True,top=True,pad=7,labelsize=23)
    ax_yx.tick_params(which='major',direction='in',width=2,length=10,right=True,top=True,pad=7,labelsize=23)
    ax_zy.tick_params(which='major',direction='in',width=2,length=10,right=True,top=True,pad=7,labelsize=23)

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

    plt.gcf().text(0.63, 0.87, r'\bf {'+name+r'}', fontsize=60)
    plt.gcf().text(0.63,0.82,str(nstars)+' stars',fontsize=30)
    plt.gcf().text(0.63,0.79,r'$\langle$[Fe/H]$\rangle$ = '+r'{:.2f}'.format(mean(feh)),fontsize=30)           
    plt.gcf().text(0.63,0.75,'$\sigma_r$ = '+'{:.1f}'.format(sigx)+' km s$^{-1}$',fontsize=30)           
    plt.gcf().text(0.63,0.72,'$\sigma_\phi$ = '+'{:.1f}'.format(sigy)+' km s$^{-1}$',fontsize=30)           
    plt.gcf().text(0.63,0.69,'$\sigma_z$ = '+'{:.1f}'.format(sigz)+' km s$^{-1}$',fontsize=30)  
    
    points = zeros(shape=(nstars,3))
    points[:,0] = x
    points[:,1] = y
    points[:,2] = z
    Sun = array([8.0,0.0,0.0])
    SunOverlap = in_hull(Sun,points)
    if SunOverlap:
        plt.gcf().text(0.63,0.66,r'Contains Sun',fontsize=30)
    else:
        plt.gcf().text(0.63,0.66,r'Does not contain Sun',fontsize=30)


    fig.savefig('../plots/stars/Vtriangle_'+name+'.pdf',bbox_inches='tight')
    plt.clf
    return fig