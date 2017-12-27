# GalfitPyWrap

A simple galfit wrapper for python. It allows the definition of models to fit, the creation of input files, and running galfit all within python. This wrapper is very simple and created to use galfit in batch mode.

## Requirements

To run you need:

* galfit (in the path)
* pyfits, numpy, matplotlib 
* (optional) sextractor (in the path): if using sxmask for initial conditions and masking

## Usage

After importing the library, you need to define the models for galfit to run as dictionaries:

```
models=[
	{0:'sersic',1:'167 167 1 1',3:'0 1',4:'20 1',5:'4 1',9:'1 1',10:'0 1','Z':0,'Comment':'Sersic 0'}
]
```

Please see the [galfit documentation](https://users.obs.carnegiescience.edu/peng/work/galfit/README.pdf) for details on how to define the models to fit.

By calling `CreateFile` you create the input file needed to run galfit. For this step you need to give the input image file, the region within the image to fit, the models. Any other parameter not explicitly passed in the function call will assume a default value defined within the function for all the galfit input file parameters.

Finally, one run galfit by passing the created file to `rungalfit`. This will create all the galfit outputs in the location of the input file, and will return galfit stdout, the open fits file output by galfit, the best fit models and an exit signal (0: success, 1: error, 124: timeout)

To see an example of this scheme please see the examples section.

### Sextractor pass

There is an option to run sextractor to mask undesired objects and to create initial condition models. This can be achieved by running sxmsk, which returns a mask, a list of models and a hierarchy tree with the objects to remove given different masking options (0: no mask, 1: central object only, 2: central and adjacent objects. This is given by nrem parameter). The sextractor configuration files and parameters files needs to be defined by the user, the only requirement is that it creates a segmentation image.