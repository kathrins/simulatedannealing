import scipy.stats as sst
import numpy as np
import scipy.spatial.distance as ssp
import matplotlib.pyplot as plt
import itertools


# SIMULATED ANNEALING



# -------------------------------------------
# DEFINE FUNCTIONS TO CALCULATE VARIOGRAM
# -------------------------------------------

# Distance Matrix
def calc_distmatrix(dataset):
    d = ssp.squareform(ssp.pdist(dataset[:,[0,1]]))
    d1 = d[np.triu_indices(d.shape[0])]
    d2 = np.triu(d,k=0)
    return d,d1,d2
# Bins
#use 0.5* maximum distance bc of small number of values for high distances
def create_bins(distancematrix):
    bins = np.linspace(0,0.5*np.max(distancematrix),num=15)
    return bins

# Variogram (assume isotropic)
def calc_variogram(dataset):
    d,d1,d2 = calc_distmatrix(dataset)
    bins = create_bins(d)
    vario = np.zeros((bins.shape[0] - 1, 2))

    for i in range(bins.shape[0]-1):
        idx = np.where(np.logical_and(d2>bins[i],d2<=bins[i+1]))
        vario[i,0] = (bins[i]+bins[i+1])/2
        if np.sum(idx) > 0: #x und y koordinate
            var = np.zeros(len(idx[0]),)
            for j in range(len(idx)):
                var[j] = np.square(dataset[[idx[0][j]],3] - dataset[[idx[1][j]],3])       #k value is in column 3 of dataset
                vario[i,1] = (np.mean(var))
        else: vario[i,1]=0
    return vario

def build_edf(measurements):
    npF1 = np.array(measurements)
    nPts = np.shape(measurements)[0]
    ErwTr = 1./( 2.*nPts )
    EDFWert = 1./( nPts )
    asort = np.arange( (ErwTr), (1.), (EDFWert) )
    ind = sst.rankdata(npF1, method='ordinal')

    StdF1 = asort[np.int_(ind)-1]

    return StdF1

# -------------------------------------------
# IMPORT AND CREATE DATA
# -------------------------------------------

# edf of real data
data = np.genfromtxt('./data/Borden3DMegaInfo.txt',
                     skip_header=1)
# select data from plane with z=1
# data columns: X,Y,Z,K
z=1
data2d = data[data[:,2] == z,:]
X=data2d[:,0]
Y=data2d[:,1]

K_edf = build_edf(data2d[:, 3])
F_K=sst.norm.ppf(K_edf)
print('F_K,edf',F_K.shape)
data2d[:,3]=F_K


# -------------------------------------------
# CHOOSE INITIAL STATE AND PARAMETERS FOR SA
# -------------------------------------------
# Select a stopping criterion

#initial solution

nx=50
ny=50
x=np.linspace(0,nx,nx+1)
y=np.linspace(0,ny,ny+1)
xy=np.array(list(itertools.product(x,y)))
zz=np.zeros(xy.shape[0])+z
randK=np.random.normal(0,1,xy.shape[0])
sol = np.concatenate((xy,zz.reshape(zz.shape[0],1),randK.reshape(randK.shape[0],1)),axis=1)

d,d1,d2 = calc_distmatrix(sol)
bins = create_bins(d)
print(bins.shape)

# annealing schedule
T0=1
t=0
alpha=0.9
T=T0*(alpha**t)
M=10
m=0
itmax=5000
T_save=[]
delta_save=[]

#set counter for total number of iterations to zero
it=0

# Calculate Objective Function f(i)
# Real data did not result in a meaningful variogram, therefore we used a theoretical gaussian variogram
#f0 = calc_variogram(data2d)                  #variogram of real data
centers = (bins[1:] + bins[0:-1])/2
sill = 1
ra = 10
gauss=sill*(1-np.exp(-np.square(centers)/ra**2)) #gaussian variogram model
f0 = np.zeros((centers.shape[0], 2))
f0[:, 0] = centers
f0[:, 1] = gauss

vario = calc_variogram(sol)
plt.plot(f0[:,0],f0[:,1],label='Original data')
plt.scatter(vario[:,0],vario[:,1],label='Random data')
plt.title('Initial Variogram')
plt.legend()
plt.show()
f=np.mean(np.square(f0-vario))
F_save=[f]
#plt.plot(f0[:,0],f0[:,1],label='Original data')
print('initial f',f)
# -------------------------------------------
# START ITERATION
# -------------------------------------------

while it<itmax:
    it=it+1
    T_save=np.append(T_save,T)
    fold=f.copy()
    
    # Generate a new solution by swapping locations of two values
    r1 = np.random.randint(len(sol))
    r2 = np.random.randint(len(sol))
    solnew = sol.copy()
    solnew[r1,3],solnew[r2,3] = sol[r2,3],sol[r1,3]


    # Update Objective Function f(j)

    vario_new=vario.copy()
    for i in range (bins.shape[0]-1):
        d_r1=d[:,r1]
        idx1=np.where(np.logical_and(d_r1>bins[i],d_r1<=bins[i+1]))
        var_old = np.zeros(len(idx1[0]),)
        var_new = np.zeros(len(idx1[0]),)
        if np.sum(idx1) > 0:
            for j in range (len(idx1)):
                var_old[j] = np.square(sol[[idx1[0][j]],3] - sol[r1,3])
                var_new[j] = np.square(solnew[[idx1[0][j]],3] - solnew[r1,3])
        if np.sum(var_old)+np.sum(var_new)>0:
            vario_new[i,1]=vario[i,1]-np.mean(var_old)+np.mean(var_new)
        else: vario_new[i,1]=vario[i,1]

        
    for i in range (bins.shape[0]-1):
        d_r2=d[:,r2]
        idx1=np.where(np.logical_and(d_r2>bins[i],d_r2<=bins[i+1]))
        var_old = np.zeros(len(idx1[0]),)
        var_new = np.zeros(len(idx1[0]),)
        if np.sum(idx1)>0:
            for j in range (len(idx1)):
                var_old[j] = np.square(sol[[idx1[0][j]],3] - sol[r2,3])
                var_new[j] = np.square(solnew[[idx1[0][j]],3] - solnew[r2,3])
        if np.sum(var_old)+np.sum(var_new)>0:
            vario_new[i,1]=vario[i,1]-np.mean(var_old)+np.mean(var_new)
        else: vario_new[i,1]=vario[i,1]

    f = np.mean(np.square(f0 - vario_new))
    F_save=np.append(F_save,f)

    delta = f-fold
    delta_save=np.append(delta_save,delta)

    if delta<0:
        sol = solnew.copy()
        vario=vario_new.copy()
    elif np.random.uniform(0,1)<np.exp(-delta/T):
        sol = solnew.copy()
        vario=vario_new.copy()


    #if it%(itmax/10)==0:  #plot progress of variogramm 10 times in the iteration process
        #lab=str(it)
        #plt.plot(vario[:, 0], vario[:, 1],label=lab)
    if m<M:
        m = m+1
    else:
        if abs(f)<1e-3:
            break
        elif T==0:
            break
        else:
            m = 0
            t = t+1
            T = T0*(alpha**t)

#plt.title('progression of variogram')
#plt.legend(bbox_to_anchor=(1,1),loc=2)
#plt.show()

print('last delta value',delta)
print('Total number of iterations',it)
print('final temperature', T)
print('final objective funcion',f)

plt.plot(f0[:,0],f0[:,1],label='Original data')
plt.scatter(vario[:,0],vario[:,1],label='Fitted variogram')
plt.title('Final Variogram')
plt.legend()
plt.show()


plt.subplot(1,2,1)
plt.title('temperature progression')
plt.plot(np.arange(it),T_save)
plt.ylabel('T')
plt.xlabel('iterations')

plt.subplot(1,2,2)
plt.plot(np.arange(it+1),F_save)
plt.xlabel('iterations')
plt.ylabel('mean square error')
plt.title('objective function ')

plt.show()

