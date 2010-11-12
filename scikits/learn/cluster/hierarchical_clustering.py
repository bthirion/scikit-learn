"""
These routines perform some hierrachical agglomerative clustering
of some input data.
Currently, only Ward's algorithm is implemented.
In the longer term, single, average and maximum linkage algorithms
should be included. 
Moreover, these should be related to a forest/tree class in the future.

Author : Bertrand Thirion, 2006-2010
"""

import numpy as np

###########################################################################
# Ward's algorithm 
###########################################################################

def euclidian_distance(x, y=None):
    """
    Considering the rows of x and y as vectors, compute the
    distance matrix between each pair of vector
    
    Parameters
    ----------
    x, array of shape (n1, p)
    y=None, array of shape (n2, p)
            if y==None, then y=x is used 

    Returns
    -------
    dist, array fo shape(n1, n2)
    """
    if y == None:
        y = x
    if x.shape[1] != y.shape[1]:
        raise ValueError, "incompatible dimension for x and y matrices"
    
    s1 = x.shape[0]
    s2 = y.shape[0]
    nx = np.reshape(np.sum(x*x, 1), (s1, 1))
    ny = np.reshape(np.sum(y*y, 1), (1, s2))
    dist = np.repeat(nx, s2, 1) + np.repeat(ny, s1, 0)
    dist -= 2*np.dot(x, y.T)
    dist = np.maximum(dist, 0)
    dist = np.sqrt(dist)
    return dist


def _inertia(i, j, moments):
    """
    Compute the variance of the set which is
    the concatenation of Feature[i] and moments[j]
    """
    n = moments[0][i] + moments[0][j]
    s = moments[1][i] + moments[1][j]
    q = moments[2][i] + moments[2][j]
    return np.sum(q - (s**2/n))


def ward(feature, verbose=0):
    """Ward clustering based on a Feature matrix

    Parameters:
    -------------
    feature: array of shape (n_samples, n_features)
             feature matrix  representing n_samples samples to be clustered
    verbose=0, verbosity level

    Returns
    -------
    #t a weightForest structure that represents the dendrogram of the data

    Caveat
    ------
    Requires n_samples to be small enough so that a matrix of shape
    (2*n_samples, 2*n_samples) to hold in memory
    """
    n_samples = feature.shape[0]
    q = 2*n_samples-1 
    if feature.ndim==1:
        feature = np.reshape(feature, (-1, 1))

    # build moments as a list
    moments = [np.zeros(q), np.zeros((q, feature.shape[1])),
                np.zeros((q, feature.shape[1]))]
    moments[0][:n_samples] = 1
    moments[1][:n_samples] = feature
    moments[2][:n_samples] = feature**2
    
    # create a inertia matrix
    inertia = np.infty * np.ones((q, q))
    for i in range(n_samples):
        for j in range(i):
            inertia[i, j] = _inertia(i, j, moments)

    # prepare the main fields
    parent = np.arange(q).astype(np.int)
    height = np.zeros(q)

    # recursive merge loop
    for k in range(n_samples, q):
        # identify the merge
        ij = inertia.argmin()
        i, j = ij/q, ij%q
        d = inertia[i, j]
        parent[i] = k
        parent[j] = k
        height[k] = d   

        # update the moments
        for p in range(3):
            moments[p][k] = moments[p][i] + moments[p][j]

        # update the inertia
        for l in range(k):
            if parent[l]==l:
                inertia[k,l] =  _inertia(l, k, moments)
        inertia[i,:] = np.infty
        inertia[j,:] = np.infty
        inertia[:,i] = np.infty
        inertia[:,j] = np.infty

    # crate the resulting tree
    #t = WeightedForest(q, parent, height)

    return parent, height





