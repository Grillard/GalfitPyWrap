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
    sh=x.shape
    theta=theta-np.pi/2
    M=np.array([(x.ravel()-x0),(y.ravel()-y0)])
    c=np.array([[np.cos(theta)/q,np.sin(theta)/q],[-np.sin(theta),np.cos(theta)]])
    M=np.abs(np.dot(c,M))**(c0+2)
    return ((M[0,:]+M[1,:])**(1/(c0+2))).reshape(sh)

def gella4(x,y,x0,y0,q,theta,a):
    #Radius of a given x,y in a generalized elliptic profile modified with third and fourth elements of a fourier series.
    # a=[a3,b3,a4,b4]
    sh=x.shape
    theta=theta-np.pi/2
    M=np.array([(x.ravel()-x0),(y.ravel()-y0)])
    c=np.array([[np.cos(theta)/q,np.sin(theta)/q],[-np.sin(theta),np.cos(theta)]])
    M=np.abs(np.dot(c,M))**2
    th=np.arctan2((y.ravel()-y0),(x.ravel()-x0)*q)
    ff=1+np.sum([(a[2*i]*np.cos((i+3)*th)+a[2*i+1]*np.sin((i+3)*th)) for i in range(2)],0)
    return (np.sqrt(M[0,:]+M[1,:])*ff).reshape(sh)

def imgfee(ee,E,sh,l0=500,intzoom=1,pick=1):
    #Create a 2D image from the output of Ellipse
    #E is the extent of the image
    #sh is the shape of the image
    riall=[]
    rival=[]
    riarr=[]
    r=np.sort(np.append(ee[0]['r'],np.linspace(0,np.max(ee[0]['r']),l0)))
    xx2,yy2=np.meshgrid(UU.middlebin(np.linspace(E[0],E[1],sh[1]*intzoom+1)),UU.middlebin(np.linspace(E[2],E[3],sh[0]*intzoom+1)))
    for rii in r:
        if pick==0:
            ri=gell(xx2,yy2,ee[2]['x0'](rii),ee[2]['y0'](rii),ee[2]['q'](rii),ee[2]['theta'](rii),0.)
        elif pick==1:
            ri=gell(xx2,yy2,ee[2]['x0'](rii),ee[2]['y0'](rii),ee[2]['q'](rii),ee[2]['theta'](rii),ee[2]['c0'](rii))
        elif pick==2:
            ri=gella4(xx2,yy2,ee[2]['x0'](rii),ee[2]['y0'](rii),ee[2]['q'](rii),ee[2]['theta'](rii),[0.,ee[2]['a'][0](rii),ee[2]['a'][0](rii),0.])
        # ri=gell(xx2,yy2,ee[2]['x0'](rii),ee[2]['y0'](rii),ee[2]['q'](rii),ee[2]['theta'](rii),tc0)
        idxs=(ri<rii)
        ri[idxs]=ee[2]['val'](rii)
        ri[~idxs]=0.
        riall.append(ri)
        rival.append(ee[2]['val'](rii))
        riarr.append(rii)
    rieff=np.max(riall,0)
    from scipy.ndimage import zoom
    rieffnew=zoom(rieff,1./intzoom)
    return rieffnew,rival,riarr

def Ellipse(sciimg,mskimg=None,ps=None,E=None,img=False,Cts=False,pick=1,extrapolate=0.,tol=1,sclip=True):
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
            # x3a=minimize(lambda x:tomin2(Cteff[:,0],Cteff[:,1],[x[0],x[1],x[2],x[3],[x[4],x[5],x[6],x[7]]],[x2b['x'][3]-np.pi/2,x2b['x'][3]+np.pi/2]),np.append(x2b['x'],[0.,0.,0.,0.]),method='Powell',options={'maxiter':100000})
            # xf=minimize(lambda x:tomin2(Cteff[:,0],Cteff[:,1],[x[0],x[1],x[2],x[3],[x[4],x[5],x[6],x[7]]],[x3a['x'][3]-np.pi/2,x3a['x'][3]+np.pi/2]),x3a['x'],method='Nelder-Mead',options={'maxiter':100000})
            # r=gella4(Cteff[:,0],Cteff[:,1],xf['x'][0],xf['x'][1],xf['x'][2],xf['x'][3],[xf['x'][4],xf['x'][5],xf['x'][6],xf['x'][7]])
            x3a=minimize(lambda x:tomin2(Cteff[:,0],Cteff[:,1],[x[0],x[1],x[2],x[3],[0.,x[4],x[5],0.]],[x2b['x'][3]-np.pi/2,x2b['x'][3]+np.pi/2]),np.append(x2b['x'],[0.,0.]),method='Powell',options={'maxiter':100000})
            xf=minimize(lambda x:tomin2(Cteff[:,0],Cteff[:,1],[x[0],x[1],x[2],x[3],[0.,x[4],x[5],0.]],[x3a['x'][3]-np.pi/2,x3a['x'][3]+np.pi/2]),x3a['x'],method='Nelder-Mead',options={'maxiter':100000})
            r=gella4(Cteff[:,0],Cteff[:,1],xf['x'][0],xf['x'][1],xf['x'][2],xf['x'][3],[0.,xf['x'][4],xf['x'][5],0.])
        y['ps'].append(p)
        y['x0'].append(xf['x'][0])
        y['y0'].append(xf['x'][1])
        y['q'].append(xf['x'][2])
        tt=xf['x'][3]
        # while tt>np.pi/2 or tt<-np.pi/2:
        #     if tt>np.pi/2:tt-=np.pi
        #     elif tt<-np.pi/2:tt+=np.pi
        y['theta'].append(tt)
        if pick==1:
            y['c0'].append(xf['x'][4])
        elif pick==2:
            # y['a4'].append(xf['x'][6])
            # y['a'].append(xf['x'][4:])
            y['a4'].append(xf['x'][5])
            y['a'].append([0.,xf['x'][4],xf['x'][5],0.])
        y['val'].append(effpit(100-p))
        y['r'].append(np.median(r))
        y['DL'].append(x2b['fun']/xf['fun'])
        y['rc2'].append(xf['fun']/len(Cteff[:,0]))
        if img or Cts:
            if pick==0:
                ri=gell(xx,yy,xf['x'][0],xf['x'][1],xf['x'][2],xf['x'][3],0.)
            elif pick==1:
                ri=gell(xx,yy,xf['x'][0],xf['x'][1],xf['x'][2],xf['x'][3],xf['x'][4])
            elif pick==2:
                # ri=gella4(xx,yy,xf['x'][0],xf['x'][1],xf['x'][2],xf['x'][3],xf['x'][4:])
                ri=gella4(xx,yy,xf['x'][0],xf['x'][1],xf['x'][2],xf['x'][3],[0.,xf['x'][4],xf['x'][5],0.])
            v=np.median(r) if np.median(r)>np.min(ri) else (np.min(ri)+np.min(ri[ri>np.min(ri)]))/2 # argh...
            Ctf1=UU.getContours(ri,v,E)
            Ctout.append([Ctf1[0],Cteff])
            # if len(y['x0'])==5:
            #     print xf['x'][0],xf['x'][1],xf['x'][2],xf['x'][3],xf['x'][4]
            #     return xx,yy,ri,E,v,Ctf1
            # ri=gell(xx,yy,x2b['x'][0],x2b['x'][1],x2b['x'][2],0.,x2b['x'][3])
            # v=np.median(r1) if np.median(r1)>np.min(ri) else (np.min(ri)+np.min(ri[ri>np.min(ri)]))/2 # argh...
            # Ctf1=UU.getContours(ri,v,E)
            # ri=gell(xx,yy,x3['x'][0],x3['x'][1],x3['x'][2],x3['x'][3],x3['x'][4])
            # v=np.median(r2) if np.median(r2)>np.min(ri) else (np.min(ri)+np.min(ri[ri>np.min(ri)]))/2 # argh...
            # Ctf2=UU.getContours(ri,v,E)
            # ri=gella4(xx,yy,x4['x'][0],x4['x'][1],x4['x'][2],x4['x'][3],x4['x'][4])
            # v=np.median(r3) if np.median(r3)>np.min(ri) else (np.min(ri)+np.min(ri[ri>np.min(ri)]))/2 # argh...
            # Ctf3=UU.getContours(ri,v,E)
            # Ctout.append([Ctf1[0],Ctf2[0],Ctf3[0]])

        if img:
            plt.imshow(sciimg,extent=E,origin='lower')
            plt.plot(Cteff[:,0],Cteff[:,1],color='red',lw=2)
            plt.plot(Ctf1[0][:,0],Ctf1[0][:,1],c='black',lw=1.)

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
    rett=[y,Ctout]
    # if Cts: rett.append(Ctout)
    ellarr2={}
    for key in keys:
        if key=='r':
            ellarr2[key]=interp1d(np.append([0],y[key]),np.append([0],y[key]))
        elif key=='a':
            ellarr2[key]=[interp1d(np.append([0],y['r']),np.append([y[key][0][i]],y[key][:,i])) for i in range(4)]
        else:
            ellarr2[key]=interp1d(np.append([0],y['r']),np.append([y[key][0]],y[key]))
    rett.append(ellarr2)
    return rett
