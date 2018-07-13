import scipy.stats as sst
import numpy as np
import scipy.spatial.distance as ssp
import matplotlib.pyplot as plt
import itertools as iter

# SIMULATED ANNEALING


data = np.genfromtxt('./data/Borden3DMegaInfo.txt',
                     skip_header=1)
# select data from plane with z=1
# data columns: X,Y,Z,K
data2d = data[data[:,2] == 1,:]
X=data2d[:,0]
Y=data2d[:,1]
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
    """builds an edf of a 1D np array"""
    npF1 = np.array(measurements)
    nPts = np.shape(measurements)[0]
    ErwTr = 1./( 2.*nPts )
    EDFWert = 1./( nPts )
    asort = np.arange( (ErwTr), (1.), (EDFWert) )
    ind = sst.rankdata(npF1, method='ordinal')

    StdF1 = asort[np.int_(ind)-1]

    return StdF1

def update_vario (sol,solnew, d,vario,r1,r2):
    vario_new=vario.copy()
    for i in range (bins.shape[0]):
        d_r1=d[:,r1]
        idx1=np.where(np.logical_and(d_r1>bins[i],d_r1<=bins[i+1]))
        var_old = np.zeros(len(idx[0]),)
        var_new = np.zeros(len(idx[0]),)
        if np.sum(idx1) > 0:
            for j in range (len(idx1)):
                var_old[j] = np.square(sol[[idx[0][j]],3] - sol[[idx[1][j]],3])
                var_new[j] = np.square(solnew[[idx[0][j]],3] - solnew[[idx[1][j]],3])
            
            vario_new[i,1]=vario[i,1]-np.sum(var_old)+np.sum(var_new)
        
    for i in range (bins.shape[0]):
        d_r2=d[:,r2]
        idx1=np.where(np.logical_and(d_r2>bins[i],d_r2<=bins[i+1]))
        var_old = np.zeros(len(idx[0]),)
        var_new = np.zeros(len(idx[0]),)
        if np.sum(idx1) > 0:
            for j in range (len(idx1)):
                var_old[j] = np.square(sol[[idx[0][j]],3] - sol[[idx[1][j]],3])
                var_new[j] = np.square(solnew[[idx[0][j]],3] - solnew[[idx[1][j]],3])
            
    vario_new[i,1]=vario[i,1]-np.sum(var_old)+np.sum(var_new)
        
    return vario_new

# -------------------------------------------
# CHOOSE INITIAL STATE AND PARAMETERS FOR SA
# -------------------------------------------
# Select a stopping criterion

#initial solution
x=np.linspace(np.min(X),np.max(X),99)
y=np.linspace(np.min(Y),np.max(Y),99)
xy=np.zeros(9999,1)
for i in range (99):
    for j in range (99):
        xy[i,0]
xy=list(iter.product(x,y))
z=np.ones(x.shape[0])

randK=np.random.normal(0,1,x.shape[0])
xy.append(z)
xy.append(randK)
sol = np.array(xy)

d,d1,d2 = calc_distmatrix(sol)
bins = create_bins(d)

#annealing schedule from Deutsch and Cockerham, Table 1 Default
# (t0, lambda, K max, K accept, S, O min)
#sched = np.array([1,0.1,100,10,3,0.001])
T0=1
t=0
alpha=0.8
T=T0*(alpha**t)
M=10
m=0
print('temperature',T)
#set counter for total number of iterations to zero
it=0
#initial value for delta to let the loop run
#delta=10
# Calculate Objective Function f(i)
f0=calc_variogram(data2d)
vario=calc_variogram(sol)
print('variogram of data2d',f0)
plt.plot(f0[:,0],f0[:,1])
plt.scatter(f1[:,0],f1[:,1])
plt.show()
f=np.mean(np.square(f0-f1))
print('initial f',f)
# -------------------------------------------
# START ITERATION
# -------------------------------------------

while it<100:
    it=it+1
    fold=f.copy()
    
    # Generate a new solution by swapping locations of two values
    r1 = np.random.randint(len(sol))
    r2 = np.random.randint(len(sol))
    solnew = sol.copy()
    solnew[r1,3],solnew[r2,3] = sol[r2,3],sol[r1,3]


    # Calculate Objective Function f(j) UPDATE
    #f=np.mean(np.square(f0-calc_variogram(solnew)))
    #f=update_vario(sol,solnew, d,vario,r1,r2)

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
        #else: var_old=0,var_new=0
        
        vario_new[i,1]=vario[i,1]-np.sum(var_old)+np.sum(var_new)
        
    for i in range (bins.shape[0]-1):
        d_r2=d[:,r2]
        idx1=np.where(np.logical_and(d_r2>bins[i],d_r2<=bins[i+1]))
        var_old = np.zeros(len(idx1[0]),)
        var_new = np.zeros(len(idx1[0]),)
        if np.sum(idx1)>0:
            for j in range (len(idx1)):
                var_old[j] = np.square(sol[[idx1[0][j]],3] - sol[r2,3])
                var_new[j] = np.square(solnew[[idx1[0][j]],3] - solnew[r2,3])
                
            #else: var_old=0,var_new=0
        vario_new[i,1]=vario[i,1]-np.sum(var_old)+np.sum(var_new)

    f=np.mean(np.square(vario_new-vario))
    delta = f-fold
    
    # if delta = f(j)-f(i) < 0:
    if delta<0: #besser/schlechter
        sol = solnew.copy()
        vario=vario_new.copy()
    elif np.random.uniform(0,1)<np.exp(-delta/T):
        sol = solnew.copy()
        vario=vario_new.copy()
    else:
        sol = sol
        vario=vario

    if m<M: #threshold for iteration/ neue temperatur
        m = m+1
    else:
        if abs(f)<0.0001:
            break
        elif T==0:
            break
        else:
            m = 0
            t = t+1
            T = T0*(alpha**t)
            #print(T)
final_f = f
plt.plot(f0[:,0],f0[:,1])
plt.scatter(vario[:,0],vario[:,1])
plt.show()
#generate a normal field for calculation of gamma
# always from one point (new/swapped)

# next iteration step
# else:
# if stopping criterion == True:
# take current solution i
# else:
# m=0
# t = t+1
# T = T(t)
# next iteration step

print('Total number of iterations',it)

print('final objective funcion',final_f)

