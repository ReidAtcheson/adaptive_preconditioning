import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt



import jax.numpy as jnp
import jax.nn as jnn
from jax import grad, jit, vmap

#g = grad(loss)(params,training_data[beg:end,:],training_inertias[beg:end])
@jit
def update(params,updates):
    return [p+u for p,u in zip(params,updates)]

#Make a random sparse-banded matrix 
#with bands in `bands1
#its diagonal shifted by `diag`
def make_banded_matrix(m,diag,bands,rng):
    subdiags=[rng.uniform(-1,1,m) for _ in bands] + [rng.uniform(-1,1,m) + diag] + [rng.uniform(-1,1,m) for _ in bands]
    offs = [-x for x in bands] + [0] + [x for x in bands]
    return sp.diags(subdiags,offs,shape=(m,m))

def make_block_precon(A,blocksize):
    A=sp.lil_matrix(A)
    m,_=A.shape
    blocks=[]
    for i in range(0,m,blocksize):
        beg=i
        end=min(i+blocksize,m)
        ids=list(range(beg,end))
        blocks.append(A[np.ix_(ids,ids)])
    return sp.block_diag(blocks)



def plain_precon_richardson(A,b,P,reltol=1e-8):
    m,_=A.shape

    x=np.zeros(m)
    r=b-A@x
    it=0
    while np.linalg.norm(r)/np.linalg.norm(b) > reltol:
        it=it+1
        #Precondition the residual
        z = P(r)
        #Find optimal step size
        Az = A@z
        tau = np.dot(r,Az)/np.dot(Az,Az)
        #Updates
        x = x + tau*z
        r = r - tau*Az
        print(f"it : {it}, res = {np.linalg.norm(r)}")
    return x






seed=23498732
rng=np.random.default_rng(seed)
m=512
diag=4
blocksize=32


A=make_banded_matrix(m,diag,[1,2,3,10,40,100],rng)
Ab=make_block_precon(A,blocksize)
luAb=spla.splu(sp.csc_matrix(Ab))


plt.spy(Ab,markersize=1)
plt.savefig("spyA.svg")

b=np.ones(m)
plain_precon_richardson(A,b,lambda x : luAb.solve(x))



