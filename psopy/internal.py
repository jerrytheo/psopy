"""
internal.py - Internal Implementation for PSO
=============================================

The actual implementation of the Particle Swarm Optimization solver.
`psopy.minimize_pso` calls the function `psopy.pswarmopt` defined here to
minimize the specified fitness function. Using this function directly allows
for a slightly faster implementation that does away with the need for the
additional recursive calls needed to wrap the constraint and objective
functions for compatibility with `scipy.optimize.minimize`.

Functions
---------

::

pswarmopt                       -- Perform constrained or unconstrained
                                   Particle Swarm Optimization

"""

import numpy as np
from scipy.spatial import distance
from scipy.optimize import OptimizeResult

_status_message = {
    'success': 'Optimization terminated successfully.',
    'maxiter': 'Maximum number of iterations has been exceeded.',
    'conviol': 'Unable to satisfy constraints at end of iterations.',
    'pr_loss': 'Desired error not necessarily achieved due to precision loss.'
}


def pswarmopt(fun, x0, confunc=None, friction=.8, max_velocity=5.,
              g_rate=.8, l_rate=.5, max_iter=1000, stable_iter=100,
              ptol=1e-6, ctol=1e-6, callback=None):
    """Internal implementation for `minimize_pso`.

    See Also
    --------
    `psopy.minimize_pso` : The SciPy compatible interface to this function. It
        includes the rest of the documentation for this function.
    `psopy.gen_confunc` : Utility function to convert SciPy style constraints
        to the form required by this function.

    Parameters
    ----------
    x0 : array_like of shape (N, D)
        Initial position to begin PSO from, where ``N`` is the number of points
        and ``D`` the dimensionality of each point. For the constrained case
        these points should satisfy all constraints.
    fun : callable
        The objective function to be minimized. Must be in the form
        ``fun(pos, *args)``. The argument ``pos``, is a 2-D array for initial
        positions, where each row specifies the position of a different
        particle, and ``args`` is a tuple of any additional fixed parameters
        needed to completely specify the function.
    confunc : callable
        The function that describes constraints. Must be of the form
        ``confunc(pos)`` that returns the constraint matrix.

    Notes
    -----
    Using this function directly allows for a slightly faster implementation
    that does away with the need for the additional recursive calls needed to
    wrap the constraint and objective functions for compatibility with Scipy.

    Examples
    --------
    These examples are identical to those laid out in `psopy.minimize_pso` and
    serve to illustrate the additional overhead in ensuring compatibility.

    >>> import numpy as np
    >>> from psopy import pswarmopt

    Consider the problem of minimizing the Rosenbrock function implemented as
    `scipy.optimize.rosen`.

    >>> from scipy.optimize import rosen
    >>> fun = lambda x: np.apply_along_axis(rosen, 1, x)

    Initialize 1000 particles and run the minimization function.

    >>> x0 = np.random.uniform(0, 2, (1000, 5))
    >>> res = pswarmopt(fun, x0, stable_iter=50)
    >>> res.x
    array([1.00000003, 1.00000017, 1.00000034, 1.0000006 , 1.00000135])

    Consider the constrained optimization problem with the objective function
    defined as:

    >>> fun = lambda x: (x[0] - 1)**2 + (x[1] - 2.5)**2
    >>> fun_ = lambda x: np.apply_along_axis(fun, 1, x)

    and constraints defined as:

    >>> cons = ({'type': 'ineq', 'fun': lambda x:  x[0] - 2 * x[1] + 2},
    ...         {'type': 'ineq', 'fun': lambda x: -x[0] - 2 * x[1] + 6},
    ...         {'type': 'ineq', 'fun': lambda x: -x[0] + 2 * x[1] + 2},
    ...         {'type': 'ineq', 'fun': lambda x: x[0]},
    ...         {'type': 'ineq', 'fun': lambda x: x[1]})

    Initializing the constraint function and feasible solutions:

    >>> from psopy import init_feasible, gen_confunc
    >>> x0 = init_feasible(cons, low=0., high=2., shape=(1000, 2))
    >>> confunc = gen_confunc(cons)

    Running the constrained version of the function:

    >>> res = pswarmopt(fun_, x0, confunc=confunc, options={
    ...     'g_rate': 1., 'l_rate': 1., 'max_velocity': 4., 'stable_iter': 50})
    >>> res.x
    array([ 1.39985398,  1.69992748])

    """
    # Initialization.
    position = np.copy(x0)
    velocity = np.random.uniform(-max_velocity, max_velocity, position.shape)

    pbest = np.copy(position)
    gbest = pbest[np.argmin(fun(pbest))]

    stable_max = 0
    stable_count = 0
    for ii in range(max_iter):
        # Store for threshold comparison.
        gbest_fit = fun(gbest[None])[0]

        # Determine the velocity gradients.
        dv_g = g_rate * np.random.uniform(0, 1) * (gbest - position)
        if confunc is not None:
            leaders = np.argmin(
                distance.cdist(position, pbest, 'sqeuclidean'), axis=1)
            dv_l = l_rate * \
                np.random.uniform(0, 1) * (pbest[leaders] - position)
        else:
            dv_l = l_rate * np.random.uniform(0, 1) * (pbest - position)

        # Update velocity and clip.
        velocity *= friction
        velocity += (dv_g + dv_l)
        np.clip(velocity, -max_velocity, max_velocity, out=velocity)

        # Update the local and global bests.
        position += velocity
        to_update = (fun(position) < fun(pbest))
        if confunc is not None:
            to_update &= (confunc(position).sum(axis=1) < ctol)

        if to_update.any():
            pbest[to_update] = position[to_update]
            gbest = pbest[np.argmin(fun(pbest))]

        # Termination criteria.
        if np.abs(gbest_fit - fun(gbest[None])[0]) < ptol:
            stable_count += 1
            if stable_count == stable_iter:
                stable_max = stable_count
                break
        else:
            if stable_count > stable_max:
                stable_max = stable_count
            stable_count = 0

        # Final callback.
        if callback is not None:
            position = callback(position)

    if confunc is not None and confunc(gbest[None]).sum() > ctol:
        status = 1
        message = _status_message['conviol']
    elif ii == max_iter:
        status = 2
        message = _status_message['maxiter']
    else:
        status = 0
        message = _status_message['success']

    result = OptimizeResult(x=gbest, fun=fun(gbest[None])[0], status=status,
                            message=message, nit=ii, nsit=stable_max,
                            success=(not status))

    if confunc is not None:
        result.cvec = confunc(gbest[None])

    return result
