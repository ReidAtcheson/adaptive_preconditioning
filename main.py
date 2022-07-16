import numpy as np
import scipy.sparse as sp
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



seed=23498732
rng=np.random.default_rng(seed)
A=make_banded_matrix(512,4,[1,2,3,10,40,100],rng)
plt.spy(A,markersize=1)
plt.savefig("spyA.svg")
