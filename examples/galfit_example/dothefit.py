import numpy as np
import matplotlib.pyplot as plt
import GalfitPyWrap

'''
    This example reproduces the galfit distributed test. It uses the gal.fits and the psf.fits files.
'''

#Define the model (as in galfit example):
model=[{
    0: 'sersic',              #  object type
    1: '48.5180 51.2800 1 1', #  position x, y
    3: '20.089 1',            #  Integrated magnitude   
    4: '5.1160 1',            #  R_e (half-light radius)   [pix]
    5: '4.2490 1',            #  Sersic index n (de Vaucouleurs n=4) 
    9: '0.7570 1',            #  axis ratio (b/a)  
    10: '-60.3690 1',         #  position angle (PA) [deg: Up=0, Left=90]
    'Z': 0                   #  output option (0 = resid., 1 = Don't subtract) 
}]

#The same as above but more compact (Note that model is a list of dictionaries):
model=[{0: 'sersic', 1: '48.5180 51.2800 1 1', 3: '20.089 1', 4: '5.1160 1', 5: '4.2490 1', 9: '0.7570 1', 10: '-60.3690 1', 'Z': 0}]

#Create the input file now:
O=GalfitPyWrap.CreateFile('gal.fits', [1,93,1,93], model,fout='galfit.input',ZP=26.563,Pimg='psf.fits',scale='0.038 0.038')

#Running galfit...
p,oimg,mods,EV=GalfitPyWrap.rungalfit('galfit.input')

#Lets have a look now...
def showme3(oimg,fignum=None):
    plt.figure(fignum)
    plt.subplot(131)
    plt.imshow(np.arcsinh(oimg[1].data/50),interpolation='none',cmap='viridis')
    plt.subplot(132)
    plt.imshow(np.arcsinh(oimg[2].data/50),interpolation='none',cmap='viridis')
    plt.subplot(133)
    plt.imshow(np.arcsinh(oimg[3].data/50),interpolation='none',cmap='viridis')

showme3(oimg)
#Lets try to fit the other galaxy at ~54 12. We just need to add the model to the list and run again...

model.append({0: 'sersic', 1: '54 12 1 1', 3: '21 1', 4: '5.1160 1', 5: '4.2490 1', 9: '0.7570 1', 10: '-60.3690 1', 'Z': 0})
O=GalfitPyWrap.CreateFile('gal.fits', [1,93,1,93], model,fout='galfit.input',ZP=26.563,Pimg='psf.fits',scale='0.038 0.038')
p,oimg,mods,EV=GalfitPyWrap.rungalfit('galfit.input')
showme3(oimg)

#Notice that we need to input significant amount of information in the form of
#initial conditions for the models. The sxmsk is designed to avoid this, as
#well as mask undesired objects. First, we need to provide sextractor
#configuration files. In the folder there are standard configuration files
#that can be used for this example.

msk,sxmods=GalfitPyWrap.sxmsk('gal.fits', 'galfitmask.sex', nrem=1, verb=True)
scifile,badpix=GalfitPyWrap.maskfiles('gal.fits', msk)
#Here the models are provided by the sextractor pass
#If no weight image is given to maskfiles, then a badpixel fits file will be generated, but it should be given as that:
O=GalfitPyWrap.CreateFile(scifile, [1,93,1,93], sxmods,fout='galfit.input',ZP=26.563,Pimg='psf.fits',scale='0.038 0.038',badmask=badpix)
p,oimg,mods,EV=GalfitPyWrap.rungalfit('galfit.input')
