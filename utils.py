#utils for running ellipse
import numpy as np

def middlebin(A):
    #Return the middle bin of an array
    return (A[1:]+A[:-1])*0.5

def getWP(D,w,p=None):
    from scipy.interpolate import interp1d
    #Gets percentile p for array D with weights w
    #if p is none, interpolator is return
    #p is given in percentage
    #Normalize
    w=np.array(w)
    w/=np.sum(w)/100
    #Sort ascending
    idxs=np.argsort(D)
    w=w[idxs]
    #Cumulative to find percentile
    C=np.cumsum(w)
    C=middlebin(np.append([0.],C))
    C[0]=0.
    C[-1]=100.

    # C=np.cumsum(w[1:])
    # C=np.append([0],C)

    # Interpolator between Cumulative Sum and value of A
    it=interp1d(C,np.array(D)[idxs])
    if p is None:
        return it
    else:
        return it(p)

def getContours(H,l,E=None):
    """
        Assumes H is well shown with plt.imshow(H,origin="lower",extent=E)
    """
    # H=np.flipud(H)
    # from matplotlib import _cntr as cntr
    from skimage import measure
    if E is None:
        E=[0,H.shape[1],0,H.shape[0]]
    locx=middlebin(np.linspace(E[0],E[1],H.shape[1]+1,endpoint=True))
    locy=middlebin(np.linspace(E[2],E[3],H.shape[0]+1,endpoint=True))
    y, x = np.mgrid[:H.shape[0], :H.shape[1]]

    contours = measure.find_contours(H, l)

    # c = cntr.Cntr(x, y, H)
    # res = c.trace(l)
    # nseg = len(res) // 2
    # segments, codes = res[:nseg], res[nseg:]
    toret=[]
    for seg in contours:
        x=np.interp(seg[:,1],range(H.shape[1]),locx)
        y=np.interp(seg[:,0],range(H.shape[0]),locy)
        toret.append(np.array([x,y]).T)
    S=np.array(toret)
    return S