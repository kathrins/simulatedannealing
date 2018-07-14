import matplotlib
print("mpl: ", matplotlib.__version__)
import numpy as np
print("np: ", np.__version__)
import scipy.stats as sst
import scipy as sp
print("sp: ", sp.__version__)
import scipy.optimize
import scipy.spatial.distance as ssp
import matplotlib.pylab as plt

# 1. Load Data
data = np.genfromtxt('xy_Lf_Cl.dat',
                     skip_header=1)
print(np.shape(data))


x=data[:,0]
y=data[:,1]
Lf=data[:,2]
Cl=data[:,3]

# 2. Empirical and Theoretical Marginal Distribution
nbins=np.arange(5,100,5)
plt.subplot(1, 3, 1)
plt.hist(Lf,bins=nbins, normed=True)
plt.title('Electric conductivity')
plt.ylabel('relative frequency')

plt.subplot(1, 3, 2)
plt.hist(Cl,bins=nbins, normed=True)
plt.title('Cl concentration')


plt.subplot(1, 3, 3)
plt.scatter(Lf,Cl)
plt.xlabel('electrical conductivity')
plt.ylabel('chloride concentration')
plt.show()

cov=np.cov(Lf,Cl)
print('Covariance:',cov)
cor=np.corrcoef(Lf,Cl)
print('Correlation Coefficient:',cor)

#rank correlation
[rho,p]=sst.spearmanr(Lf,Cl)
print('spearman rank correlation coefficient:',rho)

# 3. Fit Models to the Marginal Distributions
max=np.amax([np.amax(Lf),np.amax(Cl)])
xlin=np.linspace(0,max+10,100)

#Gaussian normal distribution
par_Lf_norm=sst.norm.fit(Lf)
print('Fitting parameters:')
print('normal Lf',par_Lf_norm)
Lf_norm_pdf=sst.norm.pdf(xlin,loc=par_Lf_norm[0],scale=par_Lf_norm[1])
Lf_norm_cdf=sst.norm.cdf(xlin,loc=par_Lf_norm[0],scale=par_Lf_norm[1])

par_Cl_norm=sst.norm.fit(Cl)
print('normal Cl',par_Cl_norm)
Cl_norm_pdf=sst.norm.pdf(xlin,loc=par_Cl_norm[0],scale=par_Cl_norm[1])
Cl_norm_cdf=sst.norm.cdf(xlin,loc=par_Cl_norm[0],scale=par_Cl_norm[1])

"""
Cl_cdf = np.interp(Cl, xlin, Cl_norm_cdf)
Lf_cdf = np.interp(Lf, xlin, Lf_norm_cdf)


plt.title('Cl cdf scatter')
plt.scatter(Cl, Cl_cdf)
plt.show()

plt.title('Lf cdf scatter')
plt.scatter(Lf, Lf_cdf)
plt.show()
"""

#Rayleigh distribution
par_Lf_rayleigh=sst.rayleigh.fit(Lf)
print('Rayleigh Lf',par_Lf_rayleigh)
Lf_rayleigh_pdf=sst.rayleigh.pdf(xlin,loc=par_Lf_rayleigh[0],scale=par_Lf_rayleigh[1])
Lf_rayleigh_cdf=sst.rayleigh.cdf(xlin,loc=par_Lf_rayleigh[0],scale=par_Lf_rayleigh[1])


par_Cf_rayleigh=sst.rayleigh.fit(Cl)
print('Rayleigh Cl',par_Cf_rayleigh)
Cl_rayleigh_pdf=sst.rayleigh.pdf(xlin,loc=par_Cf_rayleigh[0],scale=par_Cf_rayleigh[1])
Cl_rayleigh_cdf=sst.rayleigh.cdf(xlin,loc=par_Cf_rayleigh[0],scale=par_Cf_rayleigh[1])

#Bimodal normal distribution
def bimod_pdf(val,mu1,sigma1,mu2,sigma2):
    return (sst.norm.pdf(val,loc=mu1,scale=sigma1)+sst.norm.pdf(val,loc=mu2,scale=sigma2))/2

def bimod_cdf(val,mu1,sigma1,mu2,sigma2):
    return (sst.norm.cdf(val,loc=mu1,scale=sigma1)+sst.norm.cdf(val,loc=mu2,scale=sigma2))/2


xdata=np.arange(2.5,97.5,5)
[ydata,bins_bimod]=np.histogram(Lf,bins=np.arange(0,100,5),normed=True)
p0=[par_Lf_norm[0]-1,par_Lf_norm[1],par_Lf_norm[0]+1,par_Lf_norm[1]]
[par_Lf_bimod,cov_Lf_bimod]=sp.optimize.curve_fit(bimod_pdf,xdata,ydata,p0=p0,bounds=(0,np.inf))
Lf_bimod_pdf=bimod_pdf(xlin,par_Lf_bimod[0],par_Lf_bimod[1],
                       par_Lf_bimod[2] ,par_Lf_bimod[3])
Lf_bimod_cdf=bimod_cdf(xlin,par_Lf_bimod[0],par_Lf_bimod[1],
                       par_Lf_bimod[2] ,par_Lf_bimod[3])
print('Bimodal distribution parameters Lf mu1,sigma1,mu2,sigma2', par_Lf_bimod)

[ydata,bins_bimod]=np.histogram(Cl,bins=np.arange(0,100,5),normed=True)
p0=[par_Cl_norm[0]-1,par_Cl_norm[1],par_Cl_norm[0]+1,par_Cl_norm[1]]
[par_Cl_bimod,cov_Cl_bimod]=sp.optimize.curve_fit(bimod_pdf,xdata,ydata,p0=p0,bounds=(0,np.inf))
Cl_bimod_pdf=bimod_pdf(xlin,par_Cl_bimod[0],par_Cl_bimod[1],
                       par_Cl_bimod[2] ,par_Cl_bimod[3])
Cl_bimod_cdf=bimod_cdf(xlin,par_Cl_bimod[0],par_Cl_bimod[1],
                       par_Cl_bimod[2] ,par_Cl_bimod[3])
print('Bimodal distribution parameters Cl mu1,sigma1,mu2,sigma2', par_Cl_bimod)

#empirical distribution function
def build_edf(measurements,):
    """builds an edf of a 1D np array"""
    npF1 = np.array(measurements)
    nPts = np.shape(measurements)[0]
    ErwTr = 1./( 2.*nPts )
    EDFWert = 1./( nPts )
    asort = np.arange( (ErwTr), (1.), (EDFWert) )
    ind = sst.rankdata(npF1, method='ordinal')

    StdF1 = asort[np.int_(ind)-1]

    return StdF1

Lf_edf = build_edf(Lf)
Lf_sort_idx = np.argsort(Lf)

Cl_edf = build_edf(Cl)
Cl_sort_idx = np.argsort(Cl)

# Plot Histograms and fitted distributions
plt.subplot(2,2,1)
plt.hist(Lf,bins=nbins, normed=True,label='Histogram')
plt.plot(xlin,Lf_norm_pdf,label='Normal')
plt.plot(xlin,Lf_rayleigh_pdf,label='Rayleigh')
plt.plot(xlin,Lf_bimod_pdf,label='Bimodal')
plt.title('Lf PDF')
plt.legend()

plt.subplot(2,2,2)
plt.hist(Cl,bins=nbins, normed=True,label='Histogram')
plt.plot(xlin,Cl_norm_pdf,label='Normal')
plt.plot(xlin,Cl_rayleigh_pdf,label='Rayleigh')
plt.plot(xlin,Cl_bimod_pdf,label='Bimodal')


plt.title('Cl PDF')
plt.legend()

plt.subplot(2,2,3)
plt.hist(Lf,bins=nbins, normed=True,label='Histogram',cumulative=True)
plt.plot(xlin,Lf_norm_cdf,label='Normal')
plt.plot(xlin,Lf_rayleigh_cdf,label='Rayleigh')
plt.plot(xlin,Lf_bimod_cdf,label='Bimodal')
plt.plot(Lf[Lf_sort_idx], Lf_edf[Lf_sort_idx], label='edf')
plt.title('Lf CDF')
plt.legend()

plt.subplot(2,2,4)
plt.hist(Cl,bins=nbins, normed=True,label='Histogram',cumulative=True)
plt.plot(xlin,Cl_norm_cdf,label='Normal')
plt.plot(xlin,Cl_rayleigh_cdf,label='Rayleigh')
plt.plot(xlin,Cl_bimod_cdf,label='Bimodal')
plt.plot(Cl[Cl_sort_idx], Cl_edf[Cl_sort_idx], label='edf')
plt.title('Cl CDF')
plt.legend()
plt.show()

#4. Fit Bivariate Gaussian Model


xgrid, ygrid = np.mgrid[0:max:1, 0:max:1]
pos = np.empty(xgrid.shape + (2,))
pos[:, :, 0] = xgrid
pos[:, :, 1] = ygrid
rv = sst.multivariate_normal(mean=[par_Lf_norm[0],par_Cl_norm[0]],cov=cov)
plt.pcolormesh(xgrid, ygrid, rv.pdf(pos))
plt.colorbar()

plt.scatter(Lf,Cl,label='data')
plt.xlabel('electrical conductivity')
plt.ylabel('chloride concentration')
plt.legend()
plt.title('Bivariate normal distribution')
plt.show()



#The data points are arranged in two "branches". Most of the data points don't fall in area of high probability and the
#shape of the probability density does not agree with the shape of the scatter plot very well.

#Compare with EDF scatter plot
plt.subplot(1,2,1)
plt.scatter(Lf,Cl,label='data')
plt.xlabel('electrical conductivity')
plt.ylabel('chloride concentration')
plt.title('Data scatter plot')

plt.subplot(1,2,2)
plt.scatter(Lf_edf,Cl_edf,label='edf')
plt.xlabel('electrical conductivity')
plt.ylabel('chloride concentration')
plt.legend()
plt.title('EDF scatter plot')
plt.show()

#The EDF data points seem to be less concentrated  on two lines than the original data


F_Lf=sst.norm.ppf(Lf_edf)
F_Cl=sst.norm.ppf(Cl_edf)

plt.subplot(1,2,1)
plt.title('hist of F_Cl')
plt.hist(F_Cl, normed=True)
plt.subplot(1,2,2)
plt.title('hist of F_Lf')
plt.hist(F_Lf, normed=True)
plt.show()




# 5. Distance Matrix
D=ssp.squareform(ssp.pdist(data[:,[0,1]],metric='euclidean'))
plt.imshow(D)
plt.title('Distance Matrix')
plt.show()
D1=D[np.triu_indices(len(D))]
D2=np.triu(D,k=0)
plt.hist(D1,bins=20)
plt.title('Histogram of occuring distances')
plt.show()


# 6. Empirical (Semi-)Variogramm

var = np.zeros((len(D), len(D)))
for i in range(len(D)):
    for j in range(i, len(D)):
        var[i, j] = (Lf[i] - Lf[j]) ** 2.

vars_ix = np.triu_indices(var.shape[0], 0)
vars_Lf = var[vars_ix]

plt.subplot(2, 2, 1)
plt.scatter(D1, vars_Lf, s=40, facecolors='none', edgecolors='b')
plt.ylabel("variance")
plt.title('electrical conductivity variogram cloud')

var = np.zeros((len(D),len(D)))
for i in range(len(D)):
    for j in range(i, len(D)):
        var[i,j] = (Cl[i] - Cl[j])**2.

vars_ix = np.triu_indices(var.shape[0], 0)
vars_Cl = var[vars_ix]

plt.subplot(2,2,2)
plt.scatter(D1, vars_Cl, s=40, facecolors='none', edgecolors='b')
plt.xlabel("distance")
plt.ylabel("variance")
plt.title('Cl Concentration variogram cloud')



dmax=3000   #use only values with distance<dmax because data is confined
d=100       #distance class size
bins=np.arange(0,dmax,d)
var_Lf=np.zeros((len(bins)-1,))
d_mean_Lf=np.zeros((len(bins)-1,))
for i in range(0,len(bins)-1):
    ix=np.where(np.logical_and(D2>(i*d),D2<=((i+1)*d)))
    d_mean_Lf[i]=D2[ix].mean()
    var_Lf[i]=np.var(Lf[ix[0]])
plt.subplot(2,2,3)
plt.plot(d_mean_Lf,var_Lf)
plt.title('Variogram Lf')

var_Cl=np.zeros((len(bins)-1,))
d_mean_Cl=np.zeros((len(bins)-1,))
for i in range(0,len(bins)-1):
    ix=np.where(np.logical_and(D2>(i*d),D2<=((i+1)*d)))
    d_mean_Cl[i]=D2[ix].mean()
    var_Cl[i]=np.var(Cl[ix[1]])
plt.subplot(2,2,4)
plt.plot(d_mean_Cl,var_Cl)
plt.title('Variogram Cl')
plt.show()
