#Example of Ellipse fitting

import numpy as np

#Lets create some fake gaussian profile
from scipy.stats import norm
xx,yy=np.meshgrid(np.linspace(-8,8,100,endpoint=True),np.linspace(-8,8,100,endpoint=True))
h1=10.*np.exp(-0.5*(xx**2+yy**2))

#lets add some noise:
h2=norm.rvs(1.,size=10000).reshape(100,100)

#Final fake profile:
H=h1+h2

#now the fitting
from GalfitPyWrap import Ellipse as EE
ee=EE.Ellipse(H,E=E)
eee=EE.imgfee(ee,E,H.shape)

plt.figure()
plt.subplot(131)
plt.imshow(H,extent=E,origin='lower')
plt.title('Original')
plt.subplot(132)
plt.imshow(eee[0],extent=E,origin='lower')
plt.title('Model')
plt.subplot(133)
plt.imshow(H-eee[0],extent=E,origin='lower')
plt.title('Residual')

#Slightly more complex example with two overlapping profiles

xx,yy=np.meshgrid(np.linspace(-8,8,100,endpoint=True),np.linspace(-8,8,100,endpoint=True))
h1=10.*np.exp(-0.5*(xx**2+yy**2/0.5))
h2=8.*np.exp(-0.5*((xx-1)**2/0.1+(yy-1.5)**2/0.1))
h3=norm.rvs(0.1,size=10000).reshape(100,100)
H=h1+h2+h3

#The elliptical fit can only deal with one monotonically decreasing profile
#For overlappng objects, we can give a mask and fit them one by one.
#Ideally the mask will  be created with something like Sextractor segmentation masks
#in this example we will just use an ad-hoc circle for it

msk=np.ones(H.shape)
msk[h2>1]=0

#First the main blob masking the small blob:
ee1=EE.Ellipse(H,msk,E=E)
eee1=EE.imgfee(ee1,E,H.shape)
#Now we can fit the second blob on the residual image:
ee2=EE.Ellipse(H-eee1[0],E=E)
eee2=EE.imgfee(ee2,E,H.shape)

#Lets see the results
plt.figure()
plt.subplots_adjust(wspace=0,hspace=0)
plt.subplot(331)
plt.imshow(H,extent=E,origin='lower')
plt.title('Original')
plt.xticks([],[])
plt.yticks([],[])
plt.subplot(332)
plt.imshow(eee1[0]+eee2[0],extent=E,origin='lower')
plt.title('Model')
plt.xticks([],[])
plt.yticks([],[])
plt.subplot(333)
plt.imshow(H-eee1[0]-eee2[0],extent=E,origin='lower')
plt.title('Residual')
plt.xticks([],[])
plt.yticks([],[])
plt.subplot(335)
plt.imshow(eee1[0],extent=E,origin='lower')
plt.xticks([],[])
plt.yticks([],[])
plt.subplot(336)
plt.imshow(H-eee1[0],extent=E,origin='lower')
plt.xticks([],[])
plt.yticks([],[])
plt.subplot(338)
plt.imshow(eee2[0],extent=E,origin='lower')
plt.xticks([],[])
plt.yticks([],[])
plt.subplot(339)
plt.imshow(H-eee2[0],extent=E,origin='lower')
plt.xticks([],[])
plt.yticks([],[])