import numpy as np
from scipy.spatial import distance
from scipy.optimize import OptimizeResult

_status_message = {
    'success': 'Optimization terminated successfully.',
    'maxiter': 'Maximum number of iterations has been exceeded.',
    'conviol': 'Unable to satisfy constraints at end of iterations.',
    'pr_loss': 'Desired error not necessarily achieved due to precision loss.'
}


def uconmin_pso(fun, x0, friction=.8, max_velocity=5., g_rate=.8, l_rate=.5,
                max_iter=10000, stable_iter=1000, ptol=1e-5, callback=None):
    """Unconstrained minimization of the objective function using PSO.

    See Also
    --------
    `pso.minimize_pso` : The SciPy compatible interface to this function. It
        also documents the rest of the parameters.
    `pso.conmin_pso` : PSO for constrained minimization. It documents the rest
        of the keys for `OptimizeResult`.

    Returns
    -------
    result : scipy.optimize.OptimizeResult
        The optimization result represented as a `OptimizeResult` object. The
        following keys are supported,

            x : array_like
                The solution vector.
            success : bool
                Whether or not the algorithm successfully converged.
            status : int
                Termination status. 0 if successful, 1 if max. iterations
                reached, 2 if constraints cannot be satisfied.
            message : str
                Description of the cause of termination.
            nit : int
                Number of iterations performed by the swarm.
            nsit : int
                Maximum number of iterations for which the algorithm remained
                stable.
            fun : float
                Value of objective function for x.

    Notes
    -----
    Particle Swarm Optimization (PSO) [1]_ is a biologically inspired
    metaheuristic for finding the global minima of a function. It works by
    iteratively converging a population of randomly initialized solutions,
    called particles, toward a globally optimal solution. Each particle in the
    population keeps track of its current position and the best solution it has
    encountered, called ``pbest``. Each particle has an associated
    velocity used to traverse the search space. The swarm keeps track of the
    overall best solution, called ``gbest``. Each iteration of the swarm
    updates the velocity of the particle toward a weighted sum of the ``pbest``
    and ``gbest``. The velocity of the particle is then added to the position
    of the particle.

    Shi and Eberhart [2]_ describe using an inertial weight, or a friction
    parameter, to balance the effect of the global and local search. This acts
    as a limiting factor to ensure velocity does not increase, or decrease,
    unbounded.

    References
    ----------
    .. [1] Eberhart, R. and Kennedy, J., 1995, October. A new optimizer using
        particle swarm theory. In Micro Machine and Human Science, 1995.
        MHS'95., Proceedings of the Sixth International Symposium on (pp.
        39-43). IEEE.
    .. [2] Shi, Y. and Eberhart, R., 1998, May. A modified particle swarm
        optimizer. In Evolutionary Computation Proceedings, 1998. IEEE World
        Congress on Computational Intelligence., The 1998 IEEE International
        Conference on (pp. 69-73). IEEE.

    Examples
    --------
    Let us consider the problem of minimizing the Rosenbrock function
    implemented as `scipy.optimize.rosen`.

    >>> import numpy as np
    >>> from scipy.optimize import rosen

    Initialize 1000 particles and run the minimization function.

    >>> x0 = np.random.uniform(0, 2, (1000, 5))
    >>> res = uconmin_pso(rosen, x0, stable_iter=50)
    >>> res.x
    ... array([ 1.,  1.,  1.,  1.,  1.])
    """
    # Initialization.
    position = np.copy(x0)
    velocity = np.random.uniform(-max_velocity, max_velocity, position.shape)

    pbest = np.copy(position)
    gbest = pbest[np.argmin(np.apply_along_axis(fun, 1, pbest))]

    stable_max = 0
    stable_count = 0
    for ii in range(max_iter):
        # Store for threshold comparison.
        gbest_fit = fun(gbest)

        # Determine the velocity gradients.
        dv_g = g_rate * np.random.uniform(0, 1) * (gbest - position)
        dv_l = l_rate * np.random.uniform(0, 1) * (pbest - position)

        # Update velocity and clip.
        velocity *= friction
        velocity += (dv_g + dv_l)
        np.clip(velocity, -max_velocity, max_velocity, out=velocity)

        # Update the local and global bests.
        position += velocity
        to_update = (np.apply_along_axis(fun, 1, position) <
                     np.apply_along_axis(fun, 1, pbest))

        if to_update.any():
            pbest[to_update] = position[to_update]
            gbest = pbest[np.argmin(np.apply_along_axis(fun, 1, pbest))]

        # Termination criteria.
        if np.abs(gbest_fit - fun(gbest)) < ptol:
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
            position = np.apply_along_axis(callback, 1, position)

    if ii == max_iter:
        status = 1
        message = _status_message['maxiter']
    else:
        status = 0
        message = _status_message['success']

    digits = int(-np.log10(ptol))
    best = np.round(gbest, digits)
    fun_ = np.round(fun(gbest), digits)

    return OptimizeResult(
        x=best, status=status, message=message, nit=ii, nsit=stable_max,
        fun=fun_, success=(not status))


def conmin_pso(fun, x0, confunc, friction=.8, max_velocity=1., g_rate=.1,
               l_rate=.1, max_iter=1e5, stable_iter=1e4, ptol=1e-5, ctol=1e-5,
               callback=None):
    """Unconstrained minimization of the objective function using PSO.

    See Also
    --------
    `pso.minimize_pso` : The SciPy compatible interface to this function. It
        also documents the rest of the parameters.
    `pso.uconmin_pso` : PSO for unconstrained minimization. It documents the
        rest of the keys for `OptimizeResult`.
    `pso.gen_confunc` : Utility function to convert SciPy style constraints to
        the form required by this function.

    Parameters
    ----------
    confunc : callable
        The function constructed by check_constraints. Should return the
        constraint matrix.

    Returns
    -------
    result : scipy.optimize.OptimizeResult
        The optimization result represented as a OptimizeResult object. The
        following additional keys are supported for the constrained case,

            cvec : float
                The constraint vector for x.

    Notes
    -----
    .. Explain the algorithm here.

    References
    ----------
    .. Add citations here.

    Examples
    --------
    .. Use this function directly.
    """
    # Initial position and velocity.
    position = np.copy(x0)
    velocity = np.random.uniform(-max_velocity, max_velocity, position.shape)

    # Initial local best for each point and global best.
    lbest = np.copy(position)
    gbest = lbest[np.argmin(fun(lbest))]

    stable_max = 0
    stable_count = 0
    for ii in range(max_iter):
        # Store old for threshold comparison.
        gbest_fit = fun(gbest)

        # Determine the velocity gradients.
        leaders = np.argmin(
            distance.cdist(position, lbest, 'sqeuclidean'), axis=1)
        dv_g = g_rate * np.random.uniform(0, 1) * (gbest - position)
        dv_l = l_rate * np.random.uniform(0, 1) * (lbest[leaders] - position)

        # Update velocity such that |velocity| <= max_velocity.
        velocity *= friction
        velocity += (dv_g + dv_l)
        chk = (np.abs(velocity) > max_velocity)
        velocity[chk] = np.sign(velocity[chk]) * max_velocity

        # Update the local and global bests.
        position += velocity
        to_update = (np.apply_along_axis(fun, 1, position) < fun(lbest))
        if confunc:
            to_update &= (confunc(position).sum(axis=1) < ctol)

        if to_update.any():
            lbest[to_update] = position[to_update]
            gbest = lbest[np.argmin(fun(lbest))]

        # Termination criteria.
        if np.abs(gbest_fit - fun(gbest)) < ptol:
            stable_count += 1
            if stable_count == stable_iter:
                break
        else:
            if stable_count > stable_max:
                stable_max = stable_count
            stable_count = 0

        # Final callback.
        if callback is not None:
            position = np.apply_along_axis(callback, 1, position)

    if confunc(gbest).sum() < ctol:
        status = 1
        message = _status_message['conviol']
    if ii == max_iter:
        status = 2
        message = _status_message['maxiter']
    else:
        status = 0
        message = _status_message['success']

    digits = int(-np.log10(ptol))
    best = np.round(gbest, digits)
    fun_ = np.round(fun(gbest), digits)

    return OptimizeResult(
        x=best, status=status, message=message, nit=ii, nsit=stable_max,
        fun=fun_, success=(not status), cvec=confunc(gbest[None]))
