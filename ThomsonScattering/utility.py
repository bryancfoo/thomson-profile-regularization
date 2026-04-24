import jax.numpy as jnp


#Some functions for moving data around
#Or reshaping arrays into the shape needed for the forward model
#Forward model wants the shape [Nions, Nt, Nk]

def reshape_moments(Q, Nions, Nt):
    #If the input is scalar:
    if jnp.ndim(Q)==0:
        return jnp.ones((Nions, Nt))[:, :, jnp.newaxis] * Q

    # Check to see if it's only 1D, eg one ion species
    if jnp.ndim(Q) == 1:
        # If it is, reshape to (, Nt, )
        return Q[jnp.newaxis, :, jnp.newaxis]
    elif jnp.ndim(Q) == 2:
        #This is if it's in the shape (Nions, Nt) already
        #Recast as (Nions, Nt,)
        return Q[:, :, jnp.newaxis]

# This matches the shapes of a, b 1D arrays to be [Na, Nb]
# It's here because interpax can't broadcast -.-

#Pulls params out of a output.params object
def extract_params_as_array(params, var, Nindices):
    return jnp.asarray([params[f"{var}_{i}"].value for i in range(Nindices)])

def extract_all_params_as_dict(params):
    """Extract all lmfit parameters into a dict of arrays, grouped by prefix.

    Parameters named like `Te0_3` are split into prefix `Te0` and index `3`.
    Each prefix maps to a 1-D array of its values ordered by index.

    Returns
    -------
    dict mapping prefix (str) -> jnp.array of values
    """
    from collections import defaultdict
    import re

    # Group key names by their prefix (everything before the last _<digits>)
    prefix_indices = defaultdict(list)
    for key in params:
        m = re.match(r"^(.+)_(\d+)$", key)
        if m:
            prefix_indices[m.group(1)].append(int(m.group(2)))

    result = {}
    for prefix, indices in prefix_indices.items():
        Nindices = max(indices) + 1
        result[prefix] = extract_params_as_array(params, prefix, Nindices)
    return result


#Adds a profile of params to an existing lmfit Parameters object
def add_params_from_array(params, var, value_array, settings):
    Nvalues = len(value_array)
    if isinstance(settings, dict):
        settings = [settings] * Nvalues
    for i in range(Nvalues):
        params.add(var + f"_{i}", value=value_array[i], **settings[i])
    #params are mutable (I think) so no return is needed?
