import jax
from jax import vmap # vectorizing map: maps a function along array axes
import jax.numpy as jnp


def get_chunks(L, chunk_size):
    num_chunks = L // chunk_size
    chunks = [jnp.arange(i * chunk_size, (i + 1) * chunk_size) for i in range(num_chunks)]
    if L % chunk_size != 0:
        chunks.append(np.arange(L - L % chunk_size, L))
    return chunks


# fun should return tuples of arrays
def chunk_vmap(fun, array, chunk_size=4):
    L = array[0].shape[0] 
    # print('L', L)
    chunks = get_chunks(L, chunk_size)
    results = [vmap(fun)(*tuple([a[chunk] for a in array])) for chunk in chunks]
    num_results = len(results[0])
    
    # print('result shape', len(results), num_results)
    
    return tuple([jnp.concatenate([r[k] for r in results]) for k in range(num_results)])
