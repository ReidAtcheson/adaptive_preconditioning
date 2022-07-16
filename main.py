import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


from jax.experimental import sparse
import jax.numpy as jnp
import jax.nn as jnn
from jax import grad, jit, vmap

#g = grad(loss)(params,training_data[beg:end,:],training_inertias[beg:end])
@jit
def update(params,updates,step):
    return [(p0+step*u0,p1+step*u1) for (p0,p1),(u0,u1) in zip(params,updates)]

#Make a random sparse-banded matrix 
#with bands in `bands1
#its diagonal shifted by `diag`
def make_banded_matrix(m,diag,bands,rng):
    subdiags=[rng.uniform(-1,1,m) for _ in bands] + [rng.uniform(-1,1,m) + diag] + [rng.uniform(-1,1,m) for _ in bands]
    offs = [-x for x in bands] + [0] + [x for x in bands]
    return sp.diags(subdiags,offs,shape=(m,m))

#Carve out block diagonal matrix from
#input matrix for making a 
#preconditioner
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

#Create a preconditioner by blocking up
#range and putting a rank-1-update
#in each block
def make_blr(m,blocksize):
    blocks=[]
    for i in range(0,m,blocksize):
        for j in range(0,m,blocksize):
            begi=i
            endi=min(i+blocksize,m)
            begj=j
            endj=min(j+blocksize,m)

            ki=endi-begi
            kj=endj-begj

            u=jnp.zeros((ki,1))
            vt=jnp.zeros((1,kj))
            blocks.append((u,vt))
    return blocks

#Evaluate the BLR representation
def eval_blr(m,blocks,blocksize,x):
    ib=0
    out=jnp.zeros(m)
    for i in range(0,m,blocksize):
        for j in range(0,m,blocksize):
            begi=i
            endi=min(i+blocksize,m)
            begj=j
            endj=min(j+blocksize,m)

            ki=endi-begi
            kj=endj-begj
            u,vt=blocks[ib]
            if i==j:
                out = out.at[begi:endi].add(x[begi:endi] + u@(vt@x[begj:endj]))
                #out[begi:endi] = x[begi:endi] + u@(vt@x[begj:endj])
            else:
                out = out.at[begi:endi].add(u@(vt@x[begj:endj]))
                #out[begi:endi] = u@(vt@x[begj:endj])
            ib=ib+1
    return out






def plain_precon_richardson(A,b,P,reltol=1e-5):
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




def adaptive_precon_richardson(A,b,P,params,reltol=1e-5):
    m,_=A.shape
    A=sparse.BCOO.from_scipy_sparse(A)
    b=jnp.array(b)
    x=jnp.zeros(m)
    r=jnp.ones(m)

    def single_step(params,A,b,P,x):
        r=b-A@x
        #Precondition the residual
        z = P(r)
        #Find optimal step size
        Az = A@z
        tau = jnp.dot(r,Az)/jnp.dot(Az,Az)
        return (x + tau*z,r - tau*Az)

    def loss(params,r):
        return jnp.dot(r,r)

    it=0
    while jnp.linalg.norm(r)/jnp.linalg.norm(b) > reltol:
        it=it+1
        x,r = single_step(params,A,b,P,x)
        g = grad(loss)(params,r)
        step=-1e-2
        update(params,g,step)
        print(f"it : {it}, res = {jnp.linalg.norm(r)}")
    return x









seed=23498732
rng=np.random.default_rng(seed)
m=512
diag=4
blocksize=32


#Plain preconditioned richardson
A=make_banded_matrix(m,diag,[1,2,3,10,40,100],rng)
Ab=make_block_precon(A,blocksize)
luAb=spla.splu(sp.csc_matrix(Ab))
b=np.ones(m)
#plain_precon_richardson(A,b,lambda x : luAb.solve(x))
plain_precon_richardson(A,b,lambda x : x)


#Richardson with adaptive preconditioner
blr=make_blr(m,blocksize)
adaptive_precon_richardson(A,b,lambda x : eval_blr(m,blr,blocksize,x),blr)


print(blr)



