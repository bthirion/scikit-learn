"""
Testing the hierarchical clustering modules
"""

import numpy as np
from nose.tools import assert_true

from scikits.learn.cluster.hierarchical_clustering import ward


def test_ward():
    """
    """
    x = np.random.randn(10,2)
    parents, height = ward(x)
    np.testing.assert_almost_equal(height.max(), x.var(0).sum()*x.shape[0])

def test_ward_2():
    """
    """
    n, p = 10, 2
    x = np.random.randn(n, p)
    parents, height = ward(x)
    assert_true((height[n: 2*n-1]>height[n-1: 2*n-2]).all())

def test_ward_3():
    """
    """
    n = 10
    x = np.exp(np.arange(n))[:, np.newaxis]
    parents, height = ward(x)
    print parents
    assert_true((parents[1:n]==np.arange(n, 2*n-1)).all())

if __name__ == '__main__':
    import nose
    nose.run(argv=['', __file__])
