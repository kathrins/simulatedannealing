import scipy.stats as sst
import numpy as np
import scipy.spatial.distance as ssp
import matplotlib.pyplot as plt

# SIMULATED ANNEALING


data = np.genfromtxt('./data/Borden3DMegaInfo.txt',
                     skip_header=1)
# select data from plane with z=1
# data columns: X,Y,Z,K
data2d = data[data[:,2] == 1,:]
X=data2d[0,:]


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
    bins = np.linspace(0,0.5*np.max(distancematrix),num=20)
    return bins

# Variogram (assume isotropic)
def calc_variogram(dataset):
    d,d1,d2 = calc_distmatrix(dataset)
    bins = create_bins(d)
    vario = np.zeros((bins.shape[0] - 1, 2))

    for i in range(bins.shape[0]-1):
        idx = np.where(np.logical_and(d2>bins[i],d2<=bins[i+1]))
        vario[i,0] = (bins[i]+bins[i+1])/2
        var = np.zeros(len(idx),)
        for j in range(len(idx)):
            var[j] = np.square(dataset[[idx[j,0]],3] - dataset[[idx[j,1]],3])       #k value is in column 3 of dataset
        vario[i,1] = (np.mean(var))
    return vario

f0=calc_variogram(data2d)
print('variogram of data2d',f0.shape)


# -------------------------------------------
# CHOOSE INITIAL STATE AND PARAMETERS FOR SA
# -------------------------------------------
# Select a stopping criterion

#initial solution
sol=np.random.normal(0,1,len(data2d)) #?

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
f=variogram_realdata-variogram_randomfield(solution )
#f=(X[0]+X[1]+X[2]+X[3])-(sol[0]+sol[1]+sol[2]+sol[3])
print('initial f',f)
# -------------------------------------------
# START ITERATION
# -------------------------------------------
# while-loop with multiple condititions instead of stopping criterion?
while it<100 and abs(f)>0.0001 and T>0: #and abs(delta)>0.00001
    it=it+1
    fold=f

    # Generate a new solution by swapping locations of two values
    r1 = np.random.randint(len(sol))
    r2 = np.random.randint(len(sol))
    solnew = sol
    solnew[r1],solnew[r2] = sol[r2],sol[r1]


    # Calculate Objective Function f(j) UPDATE
    f=variogram_realdata-variogram_randomfield

    delta = f-fold
    # if delta = f(j)-f(i) < 0:
    if delta<0:
        sol = solnew
    elif np.random.uniform(0,1)<np.exp(-delta/T):
        sol = solnew
    else:
        sol = sol

    if m<M:
        m = m+1
    else:
        m = 0
        t = t+1
        T = T0*(alpha**t)
        #print(T)


final_f = f
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

print('final objective funciont',final_f)