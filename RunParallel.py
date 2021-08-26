import numpy as np
from astropy.cosmology import FlatLambdaCDM
from multiprocessing import Process
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
import sys
sys.path.insert(1, '../coderesources')
import fitclust2d as myfit
import priors
import fitClus2D as fc

zclus = {'a2744' : 0.308,
         'm0416' : 0.396}
pgals = {'a2744' : [1.689791e-01, 1.965115e+00, 2.0],
        'm0416' : [3.737113e-01, 1.322081e+00, 2.0]}

cluster = 'a2744'

zlens = zclus[cluster]

# load the images
imgdat = myfit.imgclass("Files/" + cluster + '_images_0.5.dat',zlens,cosmo)

# halo
halo = myfit.haloclass()
halo.load("Files/" + cluster + "halo.dat",logflags=[True,True])

# deflection distributions (simulated separately)
losdef = myfit.defclass()
losdef.load("Files/" + cluster + "_def1-los")

# set the parameters
pgal = pgals[cluster]
phalo = np.array(halo.p).flatten().tolist()
pshr = halo.pshr
pref = pgal + phalo + pshr
plabels = ['bgal', 'agal', 'alos',
 'b1', 'x1', 'y1', 'ec1', 'es1', 's1',
 'b2', 'x2', 'y2', 'ec2', 'es2', 's2',
 'b3', 'x3', 'y3', 'ec3', 'es3', 's3',
 'gc', 'gs']


#files for deflections and outbase
deffiles = ["Files/Deflections/" + cluster + "-knn-defs", "Files/Deflections/" + cluster + "-rnn-defs", "Files/Deflections/" + cluster + "-box-defs"]
outbases = ["Files/mcmc/" + cluster + "/knn", "Files/mcmc/" + cluster + "/rnn", "Files/mcmc/" + cluster + "/rnn"]
def runMC(index):
    deffile = deffiles[index]
    outbase = outbases[index]
    
    memdef = myfit.defclass()
    memdef.load(deffile)
    
    # lensmodel setup; remember to include the specific priors
    lm = myfit.lensmodel(imgdat,halo,memdef,losdef)
    lm.setprior(priors.func)

    # initialize the fit
    fit = myfit.fitclass(lm.lnP)
    
    # check one set of parameters
    tmp = lm.lnP(pref)
    print(tmp)
    print('chisq:',-2.0*tmp['chisq'])

    # optimize
    best = fit.optimize(pref,restart=5)
    print(fit.best.message)
    print(fit.best.x)
    print(fit.best.fun)
    tmp = lm.lnP(fit.best.x)
    print(tmp)
    print('chisq:',-2.0*tmp['chisq'])

    outbase = outbase

    # run MCMC
    fit.MCset(nburn=10000,nstep=5000,basename=outbase+'-mc')
    fit.MCrun()

    # make plots
    fit.MCplot(outbase+'-mc.pdf',labels=plabels,fmt='.3f',truths=pref)

    return str(index) + "completed!"




#for writing to deflection files
b0 = 0.0
a0 = 2.0
nreps = 3000
memdef = fc.defclass()
aarr = np.linspace(1.0, 2.5, 31)
imgdat2 = myfit.imgclass("Files/m0416_images.dat",0.396,cosmo)

galfiles = ["Files/GalaxyData/m0416-def-sigma.txt", "Files/GalaxyData/m0416-def-knn.txt", "Files/GalaxyData/m0416-def-rnn.txt", "Files/GalaxyData/m0416-def-box.txt"]
basenames = ["Files/Deflections/m0416-sigma-defs", "Files/Deflections/m0416-knn-defs", "Files/Deflections/m0416-rnn-defs", "Files/Deflections/m0416-box-defs"]

def makefile(i):
    galfile = galfiles[i]
    basename = basenames[i]
    gal = fc.galclass(galfile, [True, True])
    gal.scale(b0,0.5,0.1,a0,0.4,0.03)
    memdef.draw(gal, imgdat2, b0, aarr, nreps, True, basename = basename, useD = True)





# define a example function
def rand_string(length, output):
    """ Generates a random string of numbers, lower- and uppercase chars. """
    f = open("Files/testParallel", "a")
    f.write("Hello there!")
    f.close()
    
    return "hello but hi"
