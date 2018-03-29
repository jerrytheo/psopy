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
