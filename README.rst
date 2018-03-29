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

.. include:: docs/readme/installation.rst

-------------------------------------------------------------------------------
Examples
-------------------------------------------------------------------------------

Unconstrained Optimization
==========================

Consider the problem of minimizing the Rosenbrock function, implemented as
``scipy.optimize.rosen`` using a swarm of 1000 particles.

>>> import numpy as np
>>> from psopy import minimize_pso
>>> from scipy.optimize import rosen
>>> x0 = np.random.uniform(0, 2, (1000, 5))
>>> res = minimize_pso(rosen, x0, options={'stable_iter': 50})
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
>>> res = minimize_pso(fun, x0, constrainsts=cons, options={
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
.. [1] Ray, T. and Liew, K.M., 2001. A swarm with an effective information
    sharing mechanism for unconstrained and constrained single objective
    optimisation problems. In Evolutionary Computation, 2001. Proceedings of
    the 2001 Congress on (Vol. 1, pp. 75-80). IEEE.
.. [2] Eberhart, R. and Kennedy, J., 1995, October. A new optimizer using
    particle swarm theory. In Micro Machine and Human Science, 1995. MHS'95.,
    Proceedings of the Sixth International Symposium on (pp. 39-43). IEEE.
.. [3] Shi, Y. and Eberhart, R., 1998, May. A modified particle swarm
    optimizer. In Evolutionary Computation Proceedings, 1998. IEEE World
    Congress on Computational Intelligence., The 1998 IEEE International
    Conference on (pp. 69-73). IEEE.
.. [4] Lorem ipsum dolor sit amet, consectetur adipiscing elit. Integer
    volutpat feugiat imperdiet. Phasellus placerat elit nec erat tristique
    faucibus. Suspendisse at nunc odio. Nullam sagittis nunc ut sed.
