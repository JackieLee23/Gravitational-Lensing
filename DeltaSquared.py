import numpy as np
import astropy
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.coordinates import SkyOffsetFrame, ICRS

from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
import sys
import fitclust2d as myfit
import priors


fieldcenter = {'a2744' : SkyCoord('00h14m21.2s','-30d23m50.1s'),
               'm0416' : SkyCoord('4h16m8.9s','-24d4m28.7s')}
zclus = {'a2744' : 0.308,
         'm0416' : 0.396}
pgals = {'a2744' : [1.689791e-01, 1.965115e+00, 2.0],
        'm0416' : [3.737113e-01, 1.322081e+00, 2.0]}
nimg = {'a2744': 71,
        'm0416' : 95}

def run_mcmc(cluster,mcmcout,files=[],
             subsample = [],nburn=5000,nstep=10000):
    """
    Run MCMC analysis

    Parameters
    ----------
    cluster : str
        'a2744' or 'm0416'
    files : list
        Includes [imgfile, halodatfile, memdefbase, losdefbase]:
        imgfile : File containing image info in lensmodel format using sigma=0.5
        halodatfile : File containing halo info (in .dat)
        memdefbase : File containing deflection from members (leave off .npy)
        losdefbase : File containing deflection from LOS galaxies (leave off .npy)
        If empty, will use fiducial base
    mcmcout : str
        Basename for mcmc output
    sample : list
        Subset of images to use; if empty, all images will be used
    """
    if len(files) == 0:
        imgfile = cluster+'/dat/images-sig0.5.dat'
        halodatfile = cluster+'/dat/halo.dat'
        memdefbase = cluster+'/def/def1-mem-scale1'
        losdefbase = cluster+'/def/def1-los2-scale1'
    else:
        imgfile = files[0]
        halodatfile = files[1]
        memdefbase = files[2]
        losdefbase = files[3]

    zlens = zclus[cluster]

    # load the images
    imgdat = myfit.imgclass(imgfile,zlens,cosmo)
    if len(subsample) != 0:
        imgdat.subset(subsample=subsample)

    # halo
    halo = myfit.haloclass()
    halo.load(halodatfile,logflags=[True,True])

    # deflection distributions (simulated separately)
    memdef = myfit.defclass()
    memdef.load(memdefbase)
    if len(subsample) != 0:
        memdef.subset(subsample=subsample)
    losdef = myfit.defclass()
    losdef.load(losdefbase)
    if len(subsample) != 0:
        losdef.subset(subsample=subsample)

    # lensmodel setup; remember to include the specific priors
    lm = myfit.lensmodel(imgdat,halo,memdef,losdef)
    lm.setprior(priors.func)

    # initialize the fit
    fit = myfit.fitclass(lm.lnP)

    # set the parameters
    pgal = pgals[cluster]
    phalo = np.array(halo.p).flatten().tolist()
    pshr = halo.pshr
    pref = pgal + phalo + pshr
    print(pref)
    plabels = ['bgal', 'agal', 'alos',
     'b1', 'x1', 'y1', 'ec1', 'es1', 's1',
     'b2', 'x2', 'y2', 'ec2', 'es2', 's2',
     'b3', 'x3', 'y3', 'ec3', 'es3', 's3',
     'gc', 'gs']

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

    # compute Fisher matrix
    #fit.Fisher(step=0.01)
    #print(fit.grad)
    #print(fit.Fmat)
    #print(fit.Finv)

    # run MCMC
    fit.MCset(nburn=nburn,nstep=nstep,basename=mcmcout+'-mc')
    fit.MCrun()

    # make plots
    fit.MCplot(mcmcout+'-mc.pdf',labels=plabels,fmt='.3f',truths=pref)
    #fit.plot_Fisher(outbase+'-fish.pdf',nsamp=1000,labels=plabels,truths=pref)

def get_losb(cluster,parmfile=''):
    if len(parmfile) == 0:
        parmfile = cluster+'/dat/'+cluster+'-parms-2d.dat'

    oldparms = np.loadtxt(parmfile)
    losb = []
    if cluster == 'a2744':
        irange = 23+6
    if cluster == 'm0416':
        irange = 23+19
    for i in range(23,irange):
        losb.append([np.mean(oldparms.T[i]),np.std(oldparms.T[i])])

    return losb

def get_params(mem,etab,etaa,los,losb,parms,scatter=True):
    IDarr = []; params = []

    IDarr.append('pj1')
    if len(mem.shape) == 1:
        params.append([parms[0],mem[0],[1],0,0,0,parms[1]])
    else:
        params.append([parms[0],mem[0][0],mem[0][1],0,0,0,parms[1]])
        for m in mem[1:]:
            if scatter:
                logb = parms[0]-etab*0.4*m[2] + np.random.normal(0,0.1,1)[0]
                loga = parms[1]-etaa*0.4*m[2] + np.random.normal(0,0.03,1)[0]
            else:
                logb = parms[0]-etab*0.4*m[2]
                loga = parms[1]-etaa*0.4*m[2]
            IDarr.append('pj1')
            params.append([logb,m[0],m[1],0,0,0,loga])
    for i, l in enumerate(los):
        IDarr.append('pj1')
        params.append([np.random.normal(losb[i][0],losb[i][1],1)[0],l[0],l[1],0,0,0,parms[2]])

    IDarr.append('iso');IDarr.append('iso');IDarr.append('iso');IDarr.append('shr')
    params.append([parms[3],parms[4],parms[5],parms[6],parms[7],parms[8]])
    params.append([parms[9],parms[10],parms[11],parms[12],parms[13],parms[14]])
    params.append([parms[15],parms[16],parms[17],parms[18],parms[19],parms[20]])
    params.append([parms[21],parms[22]])

    return IDarr, params

def get_mags(cluster,losb,xarr,mcmcfile,N=1000,etab=0.5,etaa=0.5,files=[]):
    """
    Run MCMC analysis

    Parameters
    ----------
    losb : arr
        b values for LOS galaxies
    xarr : arr
        Grid over which magnifications are computed
    mcmcfile : str
        MCMC main output .npy file
    N : int
        Number of maps to make
    etab, etaa : float
        Values for scaling relations
    files : list
        Includes [memdatfile, losdatfile]:
        memdatfile : File containing (x y flux) for members (.dat)
        losdatfile : File containing (x y flux) for members (.dat)
        If empty, use fiducial
    """

    chain = np.load(mcmcfile)

    if len(files) == 0:
        memdatfile = cluster+'/dat/mem.dat'
        losdatfile = cluster+'/dat/los2.dat'
    else:
        memdatfile = files[0]
        losdatfile = files[1]

    mags = []
    mem=np.loadtxt(memdatfile)
    los=np.loadtxt(losdatfile)

    for d in np.random.permutation(np.arange(len(chain)))[:N]:
        IDs,params = get_params(mem,etab,etaa,los,losb,chain[d],scatter=True)
        mu = myfit.calcmag(IDs,params,np.array(xarr),logflags=[True,True])
        mags.append(mu)

    return mags

def get_paired_arrays(array1,N1,array2,N2,Npairs):
    """
    Returns given number of non-repeating pairs for two models.
    """
    if Npairs >  N1*N2:
        return 'Error: N_pairs is greater than existing pairs.'

    count = 0; pairs = []
    tmp1 = []; tmp2 = []
    while count < Npairs:
        a = np.random.randint(0,high=N1)
        b = np.random.randint(0,high=N2)
        if [a,b] not in pairs:
            pairs.append([a,b])
            count += 1
            tmp1.append(np.array(array1[a]))
            tmp2.append(np.array(array2[b]))

    #return np.log10(np.abs(np.array(array1).flatten())), np.log10(np.abs(np.array(array2).flatten()))
    return np.abs(np.array(tmp1).flatten()), np.abs(np.array(tmp2).flatten())

def do_self_comparison(data, mu_range, mu_error = 0.01, npairs = 100):
    """
    Function to compute conditional probability over realizations of one model.
    If data is a dictionary, analysis will be combined for all self comparisons.
    Returns both the distribution and median.
    data : List or array
    """
    self_comp = {}; self_median = []

    try:
        items = data.items()
    except (AttributeError, TypeError): # type != dict
        for k in mu_range:
            mus = []
            low=(1-mu_error)*k; high=(1+mu_error)*k
            b, a = get_paired_arrays(data,len(data),data,len(data),npairs)
            mus.append(b[((a>=low)&(a<=high)&(np.isnan(b)==False)&(np.isinf(a)==False)&(np.isinf(b)==False))])
            self_median.append(np.median(mus))
            self_comp.update({k:mus})

    else: # type == dict
        for k in mu_range:
            mus = []
            low=(1-mu_error)*k; high=(1+mu_error)*k
            for key1 in data.keys():
                for key2 in data.keys():
                    if key1 == key2:
                        arr1 = data[key1][0]
                        arr2 = data[key2][0]
                        b, a = get_paired_arrays(arr1, len(arr1), arr2, len(arr2), 100)
                        mus.append(b[((a>=low)&(a<=high)&(np.isnan(b)==False)&(np.isinf(a)==False)&(np.isinf(b)==False))])
            stack = np.hstack(mus)
            self_median.append(np.median(stack))
            self_comp.update({k:stack})
    return self_comp, self_median

def do_full_comparison(data, mu_range, mu_error = 0.1, npairs = 100):
    """
    Function to compute conditional probability over realizations of many models.
    Returns both the distribution and median.
    data : Should be in dict format
    """

    full_comp = {}; full_median = []
    for k in mu_range:
        mua = []
        low=(1-mu_error)*k; high=(1+mu_error)*k
        for key1 in data.keys():
            for key2 in data.keys():
                if key1 != key2:
                    arr1 = data[key1][0]
                    arr2 = data[key2][0]
                    b, a = get_paired_arrays(arr1, len(arr1), arr2, len(arr2), 100)
                    mua.append(b[((a>=low)&(a<=high)&(np.isnan(b)==False)&(np.isinf(a)==False)&(np.isinf(b)==False))])
        stack = np.hstack(mua)
        full_median.append(np.median(stack))
        full_comp.update({k:stack})

    return full_comp, full_median

def change_map_area(data, pixel_cut: int):
    """
    Changes map area of all maps in an array.
    """

    shape = int(np.sqrt(len(data[0])))

    data_cut = []
    for i in range(len(data)):
        tmp = data[i].reshape(shape,shape)[pixel_cut:shape-pixel_cut,pixel_cut:shape-pixel_cut]
        data_cut.append(tmp.flatten())

    return data_cut

def ChainCorr(chains):
    nwalk,nstep,ndim = chains.shape
    # make sure to work with zero-mean data
    dat = chains - np.mean(chains,axis=(0,1))
    # empirical covariance matrix and its inverse
    Cmat = np.cov(dat.reshape((-1,ndim)),rowvar=False)
    Cinv = np.linalg.inv(Cmat)
    # compute auto-correlations
    autocorr = []
    for i in range(nwalk):
        shiftarr,corrarr = correlation(dat[i],dat[i],Cinv)
        autocorr.append(corrarr)
    autocorr = np.array(autocorr)
    # compute cross-correlations
    crosscorr = []
    for i in range(nwalk-1):
        shiftarr,corrarr = correlation(dat[i],dat[i+1],Cinv)
        crosscorr.append(corrarr)
    crosscorr = np.array(crosscorr)
    # done
    return shiftarr,autocorr,crosscorr

#helper method
def D2statistic(dat1,dat2,scaled=True,check=False):
    # convert to column vectors if needed
    if dat1.ndim==1: dat1 = dat1[:,np.newaxis]
    if dat2.ndim==1: dat2 = dat2[:,np.newaxis]
    # check dimensions
    ndim1 = dat1.shape[1]
    ndim2 = dat2.shape[1]
    if ndim1!=ndim2:
        print('ERROR in D2statistic: samples must have the same dimension')
        return
    # key quantities
    mu1 = np.mean(dat1,axis=0)
    mu2 = np.mean(dat2,axis=0)
    dmu = mu1-mu2
    Cmat1  = np.cov(dat1,rowvar=False)
    Cmat2  = np.cov(dat2,rowvar=False)
    if Cmat1.ndim==0:
        # we need Cmat to be a 2d array even if we are working in 1 dimension
        Cmat1 = np.array([[Cmat1]])
        Cmat2 = np.array([[Cmat2]])
    Cmat12 = 0.5*(Cmat1+Cmat2)
    Cinv1  = np.linalg.inv(Cmat1 )
    Cinv2  = np.linalg.inv(Cmat2 )
    Cinv12 = np.linalg.inv(Cmat12)
    # determine what to use for Cinv
    if scaled:
        Cinv = Cinv12
    else:
        Cinv = np.eye(ndim1)
    # Compute Delta^2 in its simple form
    d1 = np.sqrt(np.trace(Cinv@Cmat1))
    d2 = np.sqrt(np.trace(Cinv@Cmat2))
    ans = dmu@Cinv@dmu + (d1-d2)**2
    # if desired, check result against explicit sums
    if check:
        if scaled:
            d11 = distance.cdist(dat1,dat1,metric='mahalanobis',VI=Cinv)
            d22 = distance.cdist(dat2,dat2,metric='mahalanobis',VI=Cinv)
            d12 = distance.cdist(dat1,dat2,metric='mahalanobis',VI=Cinv)
        else:
            d11 = distance.cdist(dat1,dat1,metric='euclidean')
            d22 = distance.cdist(dat2,dat2,metric='euclidean')
            d12 = distance.cdist(dat1,dat2,metric='euclidean')
        s11 = np.mean(d11**2)
        s22 = np.mean(d22**2)
        s12 = np.mean(d12**2)
        test = s12 - np.sqrt(s11*s22)
        print('D2statistic check: simple/explicit/diff {:.4f} {:.4f} {:.4f}'.format(ans,test,ans-test))
    # done
    return ans

#method to call
#dat1 and dat2 are simply your mcmc chains
def TwoSampleTest(dat1,dat2,scaled=True,check=False,bootstrap=0,full_output=False):
    # reference value
    Dref = D2statistic(dat1,dat2,scaled=scaled,check=check)
    # see if we are done
    if bootstrap==0: return Dref
    # run bootstrap analysis if desired to get a picture of error in delta^2
    bootdat = []
    for iboot in range(bootstrap):
        tmp1 = resample(dat1)
        tmp2 = resample(dat2)
        Dtmp = D2statistic(tmp1,tmp2,scaled=scaled)
        bootdat.append(Dtmp)
    bootdat = np.array(bootdat)
    Dsig = np.std(bootdat,axis=0)
    Dans = np.array([Dref,Dsig]).T
    if full_output:
        return Dans,bootdat
    else:
        return Dans
