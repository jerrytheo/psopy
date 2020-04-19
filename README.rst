===============================================================================
PSOPy
===============================================================================

    A python implementation of Particle Swarm Optimization.

-------------------------------------------------------------------------------
Introduction
-------------------------------------------------------------------------------

PSOPy (pronounced "Soapy") is a SciPy compatible super fast Python
implementation for Particle Swarm Optimization. The codes are tested for
standard optimization test functions (both constrained and unconstrained).

The library provides two implementations, one that mimics the interface to
``scipy.optimize.minimize`` and one that directly runs PSO. The SciPy
compatible function is a wrapper over the direct implementation, and therefore
may be slower in execution time, as the constraint and fitness functions are
wrapped.

-------------------------------------------------------------------------------
Installation
-------------------------------------------------------------------------------

GitHub
======

To install this library from GitHub,

.. code-block:: bash

    $ git clone https://github.com/jerrytheo/psopy.git
    $ cd psopy
    $ python setup.py install

In order to run the tests,

.. code-block:: bash

    $ python setup.py test

PyPI
====

This library is available on the PyPI as psopy. If you have pip installed run,

.. code-block:: bash

    $ pip install psopy

-------------------------------------------------------------------------------
Examples
-------------------------------------------------------------------------------

Unconstrained Optimization
==========================

Consider the problem of minimizing the Rosenbrock function, implemented as
``scipy.optimize.rosen`` using a swarm of 1000 particles.

>>> import numpy as np
>>> from psopy import minimize
>>> from scipy.optimize import rosen
>>> x0 = np.random.uniform(0, 2, (1000, 5))
>>> res = minimize(rosen, x0, options={'stable_iter': 50})
>>> res.x
array([1.00000003, 1.00000017, 1.00000034, 1.0000006 , 1.00000135])

Constrained Optimization
========================

Next, we consider a minimization problem with several constraints. The intial
positions for constrained optimization must adhere to the constraints imposed
by the problem. This can be ensured using the provided function
``psopy.init_feasible``. Note, there are several caveats regarding the use of
this function. Consult its documentation for more information.

>>> # The objective function.
>>> fun = lambda x: (x[0] - 1)**2 + (x[1] - 2.5)**2
>>> # The constraints.
>>> cons = ({'type': 'ineq', 'fun': lambda x:  x[0] - 2 * x[1] + 2},
...         {'type': 'ineq', 'fun': lambda x: -x[0] - 2 * x[1] + 6},
...         {'type': 'ineq', 'fun': lambda x: -x[0] + 2 * x[1] + 2},
...         {'type': 'ineq', 'fun': lambda x: x[0]},
...         {'type': 'ineq', 'fun': lambda x: x[1]})
>>> from psopy import init_feasible
>>> x0 = init_feasible(cons, low=0., high=2., shape=(1000, 2))
>>> res = minimize(fun, x0, constrainsts=cons, options={
...     'g_rate': 1., 'l_rate': 1., 'max_velocity': 4., 'stable_iter': 50})
>>> res.x
array([ 1.39985398,  1.69992748])

-------------------------------------------------------------------------------
Authors
-------------------------------------------------------------------------------

- Abhijit Theophilus (abhijit.theo@gmail.com)
- Dr\. Snehanshu Saha (snehanshusaha@pes.edu)
- Suryoday Basak (suryodaybasak@gmail.com)

-------------------------------------------------------------------------------
License
-------------------------------------------------------------------------------

| Licensed under the BSD 3-Clause License.
| Copyright 2018 Abhijit Theophilus, Snehanshu Saha, Suryoday Basak

-------------------------------------------------------------------------------
References
-------------------------------------------------------------------------------
.. [1] Theophilus, A., Saha, S., Basak, S. and Murthy, J., 2018. A Novel
    Exoplanetary Habitability Score via Particle Swarm Optimization of CES
    Production Functions. arXiv preprint arXiv:1805.08858.
.. [2] Ray, T. and Liew, K.M., 2001. A swarm with an effective information
    sharing mechanism for unconstrained and constrained single objective
    optimisation problems. In Evolutionary Computation, 2001. Proceedings of
    the 2001 Congress on (Vol. 1, pp. 75-80). IEEE.
.. [3] Eberhart, R. and Kennedy, J., 1995, October. A new optimizer using
    particle swarm theory. In Micro Machine and Human Science, 1995. MHS'95.,
    Proceedings of the Sixth International Symposium on (pp. 39-43). IEEE.
.. [4] Shi, Y. and Eberhart, R., 1998, May. A modified particle swarm
    optimizer. In Evolutionary Computation Proceedings, 1998. IEEE World
    Congress on Computational Intelligence., The 1998 IEEE International
    Conference on (pp. 69-73). IEEE.
