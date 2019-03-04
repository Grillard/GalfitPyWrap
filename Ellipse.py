#Empirical fitting procedure for galaxies
#Works by fitting ellipses to isophotes
import numpy as np
import matplotlib.pyplot as plt
import utils as UU
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from matplotlib.path import Path

def gell(x,y,x0,y0,q,theta,c0):
    #Radius of a given x,y in a generalized boxy elliptic profile
    #defined as:
    #(x-x0)**(2+c0)+((y-y0)/q)**(2+c0)=r0
    #plus a rotation with [[c,s],[-s,c]]
    #this means:
    #x=r*cos(t)**(2/(2+c0))+x0
    #y=q*r*sin(t)**(2/(2+c0))+y0
    #Plus a rotation
    #theta 0 faces west (or x>0)
    sh=x.shape
    M=np.array([(x.ravel()-x0),(y.ravel()-y0)])
    c=np.array([[np.cos(theta),np.sin(theta)],[-np.sin(theta)/q,np.cos(theta)/q]])
    M=np.abs(np.dot(c,M))**(c0+2)
    return ((M[0,:]+M[1,:])**(1/(c0+2))).reshape(sh)

def gella4(x,y,x0,y0,q,theta,a):
    #Radius of a given x,y in a generalized elliptic profile modified with third and fourth elements of a fourier series.
    # a=[a3,b3,a4,b4]
    sh=x.shape
    # th=np.arctan2(y.ravel(),x.ravel())
    M=np.array([(x.ravel()-x0),(y.ravel()-y0)])
    c=np.array([[np.cos(theta),np.sin(theta)],[-np.sin(theta)/q,np.cos(theta)/q]])
    # M=np.dot(c,M)**2
    M=np.dot(c,M)
    th=np.arctan2(M[1,:],M[0,:])
    ff=1.+np.sum([(a[2*i]*np.cos((i+3)*th)+a[2*i+1]*np.sin((i+3)*th)) for i in range(2)],0)
    return (np.sqrt(M[0,:]**2+M[1,:]**2)/ff).reshape(sh)
    # M=M**2
    # a=-np.array(a)
    # ff=1.+np.sum([(a[2*i]*np.cos((i+3)*th)+a[2*i+1]*np.sin((i+3)*th)) for i in range(2)],0)
    # return (np.sqrt(M[0,:]+M[1,:])*ff).reshape(sh)

def getell(r,x0,y0,q,theta,c0,n=100):
    #Get ellipse at given radius
    t=np.linspace(0,np.pi/2,n,endpoint=True)
    xr0=r*np.cos(t)**(2./(2.+c0))
    yr0=q*r*np.sin(t)**(2./(2.+c0))
    xr1=np.array(list(xr0)+list((-xr0)[::-1])+list((-xr0))+list(xr0[::-1]))
    yr1=np.array(list(yr0)+list(yr0[::-1])+list(-yr0)+list((-yr0)[::-1]))
    MM=np.array([(xr1.ravel()),(yr1.ravel())])
    c=np.array([[np.cos(-theta),np.sin(-theta)],[-np.sin(-theta),np.cos(-theta)]])
    MM=np.dot(c,MM)
    xr2=(MM[0,:]+x0)
    yr2=(MM[1,:]+y0)
    return xr2,yr2

def getella4(r,x0,y0,q,theta,a,n=100):
    #Get ellipse at given radius
    # a=[a3,b3,a4,b4]
    a=np.array(a)
    th=np.linspace(0,2*np.pi,n,endpoint=True)
    ff=1.+np.sum([(a[2*i]*np.cos((i+3)*th)+a[2*i+1]*np.sin((i+3)*th)) for i in range(2)],0)
    xr0=r*np.cos(th)*ff
    yr0=q*r*np.sin(th)*ff
    # xr1=np.array(list(xr0)+list((-xr0)[::-1])+list((-xr0))+list(xr0[::-1]))
    # yr1=np.array(list(yr0)+list(yr0[::-1])+list(-yr0)+list((-yr0)[::-1]))
    MM=np.array([(xr0.ravel()),(yr0.ravel())])
    c=np.array([[np.cos(-theta),np.sin(-theta)],[-np.sin(-theta),np.cos(-theta)]])
    MM=np.dot(c,MM)
    xr2=MM[0,:]+x0
    yr2=MM[1,:]+y0
    return xr2,yr2

def eeint(y):
    keys=[x for x in y if UU.isit(y[x]) and len(y[x])==len(y['r'])]
    ellarr2={}
    for key in keys:
        if key=='r':
            ellarr2[key]=interp1d(np.append([0],y[key]),np.append([0],y[key]))
        elif key=='a':
            ellarr2[key]=[interp1d(np.append([0],y['r']),np.append([y[key][0][i]],y[key][:,i])) for i in range(4)]
        else:
            ellarr2[key]=interp1d(np.append([0],y['r']),np.append([y[key][0]],y[key]))
    return ellarr2

def imgfee(ee,E,sh,l0=500,intzoom=1,pick=1):
    #Create a 2D image from the output of Ellipse
    #E is the extent of the image
    #sh is the shape of the image
    #ee is the interpolated profiles returned by ellipse

    riall=[]
    r=np.sort(np.append(ee['r'].x,np.linspace(0,np.max(ee['r'].x),l0)))
    xx2,yy2=np.meshgrid(UU.middlebin(np.linspace(E[0],E[1],sh[1]*intzoom+1)),UU.middlebin(np.linspace(E[2],E[3],sh[0]*intzoom+1)))
    Pxy=np.array([xx2.ravel(),yy2.ravel()]).T
    rieff=np.zeros(xx2.shape)
    for rii in r:
        if pick==0:
            e=getell(ee['r'](rii),ee['x0'](rii),ee['y0'](rii),ee['q'](rii),ee['theta'](rii),0.)
            idxs0=(xx2<np.max(e[0]))&(xx2>np.min(e[0]))&(yy2<np.max(e[1]))&(yy2>np.min(e[1]))
            ri=gell(xx2[idxs0],yy2[idxs0],ee['x0'](rii),ee['y0'](rii),ee['q'](rii),ee['theta'](rii),0.)
        elif pick==1:
            e=getell(ee['r'](rii),ee['x0'](rii),ee['y0'](rii),ee['q'](rii),ee['theta'](rii),ee['c0'](rii))
            idxs0=(xx2<np.max(e[0]))&(xx2>np.min(e[0]))&(yy2<np.max(e[1]))&(yy2>np.min(e[1]))
            ri=gell(xx2[idxs0],yy2[idxs0],ee['x0'](rii),ee['y0'](rii),ee['q'](rii),ee['theta'](rii),ee['c0'](rii))
        elif pick==2:
            e=getella4(ee['r'](rii),ee['x0'](rii),ee['y0'](rii),ee['q'](rii),ee['theta'](rii),[0.,ee['a'][0](rii),ee['a'][0](rii),0.])
            idxs0=(xx2<np.max(e[0]))&(xx2>np.min(e[0]))&(yy2<np.max(e[1]))&(yy2>np.min(e[1]))
            ri=gella4(xx2[idxs0],yy2[idxs0],ee['x0'](rii),ee['y0'](rii),ee['q'](rii),ee['theta'](rii),[0.,ee['a'][0](rii),ee['a'][0](rii),0.])
        idxs=(ri<rii)
        ri[idxs]=ee['val'](rii)
        ri[~idxs]=0.
        rieff[idxs0]=np.max([ri,rieff[idxs0]],0)
        riall.append([ri,idxs0])
    # for rii in r:
    #     if pick==0:
    #         ri=gell(xx2,yy2,ee['x0'](rii),ee['y0'](rii),ee['q'](rii),ee['theta'](rii),0.)
    #     elif pick==1:
    #         ri=gell(xx2,yy2,ee['x0'](rii),ee['y0'](rii),ee['q'](rii),ee['theta'](rii),ee['c0'](rii))
    #     elif pick==2:
    #         ri=gella4(xx2,yy2,ee['x0'](rii),ee['y0'](rii),ee['q'](rii),ee['theta'](rii),[0.,ee['a'][0](rii),ee['a'][0](rii),0.])
    #     # ri=gell(xx2,yy2,ee['x0'](rii),ee['y0'](rii),ee['q'](rii),ee['theta'](rii),tc0)
    #     idxs=(ri<rii)
    #     ri[idxs]=ee['val'](rii)
    #     ri[~idxs]=0.
    #     riall.append(ri)
    # rieff=np.max(riall,0)
    #Curve of Growth
    cog={'r':r,'cl':[],'A':[]}#,'idxs':[]
    for rii in riall:
        ###
        idxs=(rii[0]!=0)
        cog['cl'].append(np.sum(rieff[rii[1]][idxs]))
        cog['A'].append(np.sum(idxs))
        # cog['idxs'].append(idxs)
    for key in cog:cog[key]=np.array(cog[key])
    from scipy.ndimage import zoom
    rieffnew=zoom(rieff,1./intzoom)
    return rieffnew,cog

def Ellipse(sciimg,mskimg=None,ps=None,E=None,img=False,pick=1,extrapolate=0.,tol=1,sclip=True):
    # sciimg is a 2D array
    # pick is 0 for elliptical profile, 1 for c0 profile, and 2 for a4 profile
    # retfun returns a linear interpolation function of the profiles
    # Extrapolate is the sky level until which one wants to extrapolate after ellipse cannot measure it anymore
    if E is None:
        E=[0,sciimg.shape[1],0,sciimg.shape[0]]
    xx,yy=np.meshgrid(UU.middlebin(np.linspace(E[0],E[1],sciimg.shape[1]+1)),UU.middlebin(np.linspace(E[2],E[3],sciimg.shape[0]+1)))
    dx=xx[0][1]-xx[0][0]
    dy=yy[1][0]-yy[0][0]
    if mskimg is None:
        mskimg=np.ones(sciimg.shape)
    pit=UU.getWP(sciimg[mskimg==1],sciimg[mskimg==1])
    # idx0=[np.where(mskimg==1)[0][np.argmax(sciimg[mskimg==1])],np.where(mskimg==1)[1][np.argmax(sciimg[mskimg==1])]]
    y={} # Summary info [ps,<x0>,<y0>,<q>,c0,a4,<theta>]
    Ctout=[]
    # for key in ['ps','x0','y0','q','theta','x0_m','y0_m','q_m','theta_m','c0','a4','val','r1','r2','r3','DL']: #,'Ct'
    for key in ['ps','x0','y0','q','theta','val','r','DL','rc2']:
        y[key]=[]
    if pick==1:
        y['c0']=[]
    elif pick==2:
        y['a4']=[]
        y['a']=[]
    if ps is None:
        #Finding the points based on the pit curve
        def safepit(x):
            if x<0: return effpit(0)
            elif x>100: return effpit(100)
            else: return np.log10(effpit(x))
        ##Finding the gradient##
        #To find a better description of the gradient, first I simplify y
        idxs=(pit.x>0) & (pit.x<=100)
        effy,b=np.unique(np.round(np.log10(pit.y[idxs]),3),return_inverse=True)
        effx=np.array([np.min(pit.x[idxs][b==i]) for i in range(len(effy))])
        effpit=interp1d(effx,10.**effy,bounds_error=False,fill_value='extrapolate') #pit is affected by the points removed above, it is very rare but it happens
        # return pit,effx,effy
        #The above is slow, naive, and arbitrary
        #Analytic approximation of the log10 pit curve
        ly=np.polyfit(np.append([effx[-1]+i*(effx[-1]-effx[-2]) for i in range(5)],effx),np.append([effy[-1]-i*(effy[-2]-effy[-1]) for i in range(5)],effy),10)
        gr=np.poly1d(np.polyder(ly))

        ps=[float(effx[-1])]
        # gdown=lambda a,x:np.abs(safepit(x-a*gr(x))-(safepit(x)+a*gr(x)))
        gdown=lambda a,x:np.abs(safepit(x-a*gr(x))-(safepit(x)+a*gr(x))) if a>0 else 1e100
        pk=1.
        # tol=1
        while ps[-1]>0:
            # print ps[-1]
            # efftol=tol/gr(ps[-1]) #absolute value
            # efftol=tol #gradient times value
            # efftol=tol/gr(ps[-1])/safepit(ps[-1]) #relative to current value
            efftol=tol/gr(ps[-1])/ps[-1] #relative to current value - nani?
            # if gr(ps[-1])<0:ps.append(pit.x[pit.x<ps[-1]][-1])
            # print ps[-1]
            # print np.min(effx)
            # print effx[effx<ps[-1]]
            # print '##############################################'
            if gr(ps[-1])<0:
                tap=0. if len(effx[effx<ps[-1]])==0 else effx[effx<ps[-1]][-1]
                ps.append(tap)
            else:
                x1=minimize(lambda x:np.abs(gdown(x,ps[-1])-efftol),pk,tol=1e-3,method='Nelder-Mead')
                pk=x1['x'][0]
                ps.append(ps[-1]-x1['x'][0]*gr(ps[-1]))
            # print pk,ps[-1]



        ############################
        # plt.clf()
        # plt.plot(pit.x,np.log10(pit.y))
        # plt.plot(pit.x,np.poly1d(ly)(pit.x))
        # plt.scatter(pit.x,np.log10(pit.y),c=gr(pit.x))
        # plt.scatter(ps,poly1d(ly)(ps),c='red')
        ############################

        ps=100-np.array(ps)
        # ps=np.append(interp1d(range(len(ps)),ps)(np.arange(0,len(ps)-1,1./pssamp)),ps[-1])
        ps=ps[ps<=100]

    pCt=None
    minmax=lambda x: np.max(x)-np.min(x)
    scieff=np.zeros(sciimg.shape)+sciimg
    scieff[mskimg==0]=np.nan
    for pi,p in enumerate(ps):
        if p==ps[0]:
            effp1=(p+ps[1])/2
            effp2=((100.-effpit.x[effpit.x.searchsorted(100.-p)])+np.min([(100.-effpit.x[effpit.x.searchsorted(100.-p)-1]),ps[pi+1]]))/2
            effp=np.min([effp1,effp2])
        else:
            if 100-p in effpit.x and pi+1!=len(ps):
                effp=((100.-effpit.x[effpit.x.searchsorted(100.-p)])+np.min([(100.-effpit.x[effpit.x.searchsorted(100.-p)-1]),ps[pi+1]]))/2
                # print (100.-pit.x[pit.x.searchsorted(100.-p)]),(100.-pit.x[pit.x.searchsorted(100.-p)-1]),100.-ps[pi+1]
            else:
                effp=p
            # Ct=UU.getContours(scieff,pit(100-effp),E)
        Ct=UU.getContours(scieff,effpit(100-effp),E)
        # Ct=[C for C in Ct if C[0][0]==C[-1][0] and C[0][1]==C[-1][1]] # only closed contours
        Ct=[np.array([x for x in C if not np.isnan(x).any()]) for C in Ct]#Remove nans
        Ct=[C for C in Ct if len(C)!=0]#Remove nans
        Ct=[C for C in Ct if np.min(C[:,0])>E[0]+1. and np.min(C[:,1])>E[2]+1. and 
            np.max(C[:,0])<E[1]-1 and np.min(C[:,0])<E[3]-1] # only Contours away from border
        if pCt is not None:
            Ct=[C for C in Ct if Path(C).contains_points([np.median(pCt,0)]).all()] # Should contain the previous one
        if len(Ct)==0:
            print 'Finishing here, you hit the background'
            break
        # Ct=[C for C in Ct if minmax(C[:,0])>dx and minmax(C[:,1])>dy] # Contours should be at least the size of a pixel
        ict=np.argmax([len(C) for C in Ct]) #picking the longest contour
        Cteff=Ct[ict]
        pCt=Cteff
        def tomin(x,y,X,T):
            #T gives the limits of the angle range
            x0,y0,q,theta,c0=np.array(X).astype(float)
            if q>1. or q<0.: return 1e100
            if c0>5 or c0<-1.9: return 1e100
            if x0>np.max(x) or x0<np.min(x): return 1e100
            if y0>np.max(y) or y0<np.min(y): return 1e100
            if theta>T[1] or theta<T[0]: return 1e100
            r=gell(x,y,x0,y0,q,theta,c0)
            rm=np.median(r)
            if rm>5*(np.max([np.max(x)-np.min(x),np.max(y)-np.min(y)])): return 1e100
            return np.sum((r-np.median(r))**2)
        def tomin2(x,y,X,T):
            x0,y0,q,theta=np.array(X[:-1]).astype(float)
            a=X[-1]
            if (np.abs(np.array(a))>0.1).any(): return 1e100
            if q>1. or q<0.: return 1e100
            if x0>np.max(x) or x0<np.min(x): return 1e100
            if y0>np.max(y) or y0<np.min(y): return 1e100
            if theta>T[1] or theta<T[0]: return 1e100
            r=gella4(x,y,x0,y0,q,theta,a)
            return np.sum((r-np.median(r))**2)
        x00=np.median(Cteff[:,0])
        y00=np.median(Cteff[:,1])
        #First guess
        t0=[]
        t1=[]
        t2=[]
        for theta in np.linspace(-np.pi/2,np.pi/2,100):
            theta=theta-np.pi/2
            M1=np.array([(Cteff[:,0]-x00),(Cteff[:,1]-y00)])
            c=np.array([[np.cos(theta),np.sin(theta)],[-np.sin(theta),np.cos(theta)]])
            M2=np.dot(c,M1)
            t0.append(theta)
            # t1.append(np.max([np.max(M2[0,:])-np.min(M2[0,:]),np.max(M2[1,:])-np.min(M2[1,:])]))
            t1.append(np.max(M2[0,:])-np.min(M2[0,:]))
            t2.append((np.max(M2[1,:])-np.min(M2[1,:]))/(np.max(M2[0,:])-np.min(M2[0,:])))
        ############
        q0=t2[np.argmax(t1)] if t2[np.argmax(t1)]<1. else 1.
        ###########
        #Sigma clipping
        if sclip and len(Cteff)>10:
            thr=np.arctan2((Cteff[:,1].ravel()-y00),(Cteff[:,0].ravel()-x00)*q0)
            rr=gell(Cteff[:,0],Cteff[:,1],x00,y00,q0,t0[np.argmax(t1)],0.)
            ly=np.polyfit(thr,rr,5)
            rf=rr-np.poly1d(ly)(thr)
            rf=(rf-np.median(rf))/np.std(rf)
            idxs=np.where(np.abs(rf)>3)[0]
            grps=np.split(idxs,np.where(np.diff(idxs)!=1)[0]+1)
            ctmsk=np.ones(Cteff[:,0].shape).astype(bool)
            for g in grps:
                if len(g)==0: continue
                g0=np.min(g)-np.argmin((rf[:np.min(g)]>1)[::-1]) if np.min(g)!=0 else 0
                g1=np.max(g)+np.argmin((rf[np.max(g):]>1)) if np.max(g)!=len(Cteff)-1 else len(Cteff)-1
                ctmsk[g0:g1]=0
            Cteff=Cteff[ctmsk,:]
        ###########
        x2a=minimize(lambda x:tomin(Cteff[:,0],Cteff[:,1],[x[0],x[1],x[2],x[3],0.],[t0[np.argmax(t1)]-np.pi/2,t0[np.argmax(t1)]+np.pi/2]),[x00,y00,q0,t0[np.argmax(t1)]],method='Powell',options={'maxiter':100000})
        x2b=minimize(lambda x:tomin(Cteff[:,0],Cteff[:,1],[x[0],x[1],x[2],x[3],0.],[x2a['x'][3]-np.pi/2,x2a['x'][3]+np.pi/2]),x2a['x'],method='Nelder-Mead',options={'maxiter':100000})
        if pick==0:
            xf=x2b
            r=gell(Cteff[:,0],Cteff[:,1],x2b['x'][0],x2b['x'][1],x2b['x'][2],x2b['x'][3],0.)
        elif pick==1:
            x3a=minimize(lambda x:tomin(Cteff[:,0],Cteff[:,1],[x[0],x[1],x[2],x[3],x[4]],[x2b['x'][3]-np.pi/2,x2b['x'][3]+np.pi/2]),np.append(x2b['x'],[0.]),method='Powell',options={'maxiter':100000})
            xf=minimize(lambda x:tomin(Cteff[:,0],Cteff[:,1],[x[0],x[1],x[2],x[3],x[4]],[x3a['x'][3]-np.pi/2,x3a['x'][3]+np.pi/2]),np.append(x3a['x'],[0.]),method='Nelder-Mead',options={'maxiter':100000})
            r=gell(Cteff[:,0],Cteff[:,1],xf['x'][0],xf['x'][1],xf['x'][2],xf['x'][3],xf['x'][4])
        elif pick==2:
            x3a=minimize(lambda x:tomin2(Cteff[:,0],Cteff[:,1],[x[0],x[1],x[2],x[3],[0.,x[4],x[5],0.]],[x2b['x'][3]-np.pi/2,x2b['x'][3]+np.pi/2]),np.append(x2b['x'],[0.,0.]),method='Powell',options={'maxiter':100000})
            xf=minimize(lambda x:tomin2(Cteff[:,0],Cteff[:,1],[x[0],x[1],x[2],x[3],[0.,x[4],x[5],0.]],[x3a['x'][3]-np.pi/2,x3a['x'][3]+np.pi/2]),x3a['x'],method='Nelder-Mead',options={'maxiter':100000})
            r=gella4(Cteff[:,0],Cteff[:,1],xf['x'][0],xf['x'][1],xf['x'][2],xf['x'][3],[0.,xf['x'][4],xf['x'][5],0.])
        y['ps'].append(p)
        y['x0'].append(xf['x'][0])
        y['y0'].append(xf['x'][1])
        y['q'].append(xf['x'][2])
        tt=xf['x'][3]
        y['theta'].append(tt)
        if pick==1:
            y['c0'].append(xf['x'][4])
        elif pick==2:
            y['a4'].append(xf['x'][5])
            y['a'].append([0.,xf['x'][4],xf['x'][5],0.])
        y['val'].append(effpit(100-p))
        y['r'].append(np.median(r))
        y['DL'].append(x2b['fun']/xf['fun'])
        y['rc2'].append(xf['fun']/len(Cteff[:,0]))
        Ctout.append(Cteff)
        if img:
            if pick==0:
                ri=gell(xx,yy,xf['x'][0],xf['x'][1],xf['x'][2],xf['x'][3],0.)
                Ctf1=getell(np.median(r),xf['x'][0],xf['x'][1],xf['x'][2],xf['x'][3],0.)
            elif pick==1:
                ri=gell(xx,yy,xf['x'][0],xf['x'][1],xf['x'][2],xf['x'][3],xf['x'][4])
                Ctf1=getell(np.median(r),xf['x'][0],xf['x'][1],xf['x'][2],xf['x'][3],xf['x'][4])
            elif pick==2:
                ri=gella4(xx,yy,xf['x'][0],xf['x'][1],xf['x'][2],xf['x'][3],[0.,xf['x'][4],xf['x'][5],0.])
                Ctf1=getella4(np.median(r),xf['x'][0],xf['x'][1],xf['x'][2],xf['x'][3],[0.,xf['x'][4],xf['x'][5],0.])
            plt.imshow(np.arcsinh(sciimg),extent=E,origin='lower')
            plt.plot(Cteff[:,0],Cteff[:,1],color='red',lw=2)
            plt.plot(Ctf1[0],Ctf1[1],c='black',lw=1.)

    y['val0']=np.max(sciimg[mskimg==1])
    keys=['r','val','x0','y0','theta','q']
    if pick==1: keys.append('c0')
    elif pick==2: keys.append('a4');keys.append('a')
    if extrapolate is not None:
        #Extrapolate after I cannot measure ellipse
        if pick==0:
            rl=gell(xx,yy,y['x0'][-1],y['y0'][-1],y['q'][-1],y['theta'][-1],0)
        if pick==1:
            rl=gell(xx,yy,y['x0'][-1],y['y0'][-1],y['q'][-1],y['theta'][-1],y['c0'][-1])
        if pick==2:
            rl=gella4(xx,yy,y['x0'][-1],y['y0'][-1],y['q'][-1],y['theta'][-1],y['a'][-1])
        dr=np.max([np.abs(y['r'][-2]-y['r'][-1]),2*(xx[0][1]-xx[0][0])])
        rmax=y['r'][-1]+dr
        pvalm=1e100
        pvn=1e100
        while True:
            valm=np.median(sciimg[(mskimg==1) & (rl<rmax) & (rl>rmax-dr)]) #Minimize
            n=len(np.where((mskimg==1) & (rl<rmax) & (rl>rmax-dr))[0])
            if valm>extrapolate and (np.sqrt(valm)/n-pvn)/pvn<1.:
                for key in keys:
                    if key=='r': y['r'].append(rmax-dr/2)
                    elif key=='val': y['val'].append(valm)
                    else: y[key].append(y[key][-1])
            else:
                valm=lambda rmax:np.abs(np.median(sciimg[(mskimg==1) & (rl<rmax) & (rl>rmax-dr)])-extrapolate) #Minimize
                xv=minimize(lambda x:valm(x) if valm(x)==valm(x) and x>=y['r'][-1] else 1e100,[rmax],method='Nelder-Mead')
                if valm(xv['x'][0])==valm(xv['x'][0]):
                    for key in keys:
                        if key=='r': y['r'].append(xv['x'][0]-dr/2)
                        elif key=='val': y['val'].append(valm(xv['x'][0]))
                        else: y[key].append(y[key][-1])
                break
            pvalm=valm
            pvn=np.sqrt(valm)/n
            rmax+=dr
    for key in y:y[key]=np.array(y[key])
    #Transform theta to common angle
    tt=np.median(y['theta'])
    y['theta'][y['theta']>tt+np.pi/2]-=np.pi
    y['theta'][y['theta']<tt-np.pi/2]+=np.pi
    rett=[y,Ctout,eeint(y)]
    # if Cts: rett.append(Ctout)
    # ellarr2=
    # rett.append(ellarr2)
    return rett
