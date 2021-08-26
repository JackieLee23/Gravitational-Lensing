import time
import numpy as np
import random

data = [1, 2, 3, 4, 5]

def setdata(arr):
    global global_data
    global_data = arr
    
def hello():
    return "hello"

def log_prob(theta):
    
    return theta[0] * theta[1]


def original_prob(theta):
    global data
    t = time.time() + np.random.uniform(0.005, 0.008)
        
    while True:
        if time.time() >= t:
            break
    print(global_data)
    global_data[0] = 55

    return -0.5 * np.sum(theta ** 2) * np.sum(data)


def numpy_test(theta):
    return -0.5 * np.sum(theta ** 2)

def setlm(lmin):
    global lm
    lm = lmin


def lnP(p):

    # here is the dictionary structure we will fill with log-likelihood
    lnPdict = {'posterior':-np.inf, 'chisq':-np.inf, 'norm':-np.inf, 'prior':-np.inf, 'chipri':-np.inf}

    # update the parameters
    lm.setp(p)

    # get galaxy quantities
    galOK,agal,Cgal,Ggal = lm.galdef.rescale(lm.bgal,lm.agal)
    # get los quantities, if appropriate
    if lm.losdef!=None:
        losOK,alos,Clos,Glos = lm.losdef.rescale(lm.bgal,lm.alos)
    else:
        # there is no los component, so set arrays to 0
        losOK = True
        alos = 0.0*agal
        Clos = 0.0*Cgal
        Glos = 0.0*Ggal

    # check whether parameters are in bounds
    if galOK==False or losOK==False:
            return lnPdict['posterior']

    # get halo contributions; remember to include D factors
    ahalo,Ghalo = lm.halo.calcdef(lm.img)
    ahalo = ahalo*np.reshape(lm.img.Darr,(lm.img.nimg,1))
    Ghalo = Ghalo*np.reshape(lm.img.Darr,(lm.img.nimg,1,1))

    # combine
    atot = ahalo + agal + alos
    Gtot = Ghalo + Ggal + Glos
    minv = Gam2minv(Gtot)
    # in test mode, minv is the identity
    #if testmode: minv = np.identity(len(minv))

    # the linear algebra requires some care; revised CRK 2021/2/6
    Ceff = dotABAT(minv,lm.img.Cmat) + Cgal + Clos
    UTSU = np.dot(lm.img.Umat.T,np.linalg.solve(Ceff,lm.img.Umat))
    darr = lm.img.xarr - atot
    dtmp = darr.flatten()
    UTSd = np.dot(lm.img.Umat.T,np.linalg.solve(Ceff,dtmp))

    # compute chisq and normalization factor
    chi_term1 = np.dot(dtmp,np.linalg.solve(Ceff,dtmp))
    chi_term2 = np.dot(UTSd,np.linalg.solve(UTSU,UTSd))
    chisq = chi_term1 - chi_term2
    s1,ldet1 = np.linalg.slogdet(UTSU)
    s2,ldet2 = np.linalg.slogdet(Ceff)
    s3,ldet3 = np.linalg.slogdet(minv)
    lnPchi = -0.5*chisq
    lnPnorm = -0.5*(ldet1+ldet2) + ldet3

    # prior
    lnPpri = lm.prior()

    # combined
    lnPtot = lnPchi + lnPnorm + lnPpri

    lnPdict['posterior'] = lnPtot
    lnPdict['chisq']     = lnPchi
    lnPdict['norm']      = lnPnorm
    lnPdict['prior']     = lnPpri
    lnPdict['chipri']    = lnPchi + lnPpri
    return lnPdict['posterior']


def Gam2minv(Garr):
    nimg = len(Garr)
    Gmat = np.zeros((2*nimg,2*nimg))
    for iimg in range(nimg):
        for i in range(2):
            for j in range(2):
                Gmat[2*iimg+i,2*iimg+j] = Garr[iimg,i,j]
    minv = np.identity(2*nimg) - Gmat
    return minv

######################################################################
# the matrix product A.B.AT comes up several times
######################################################################

def dotABAT(A,B):
    return np.linalg.multi_dot((A,B,A.T))

