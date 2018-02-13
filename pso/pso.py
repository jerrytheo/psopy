import numpy as np
from scipy.spatial import distance
from scipy.optimize import OptimizeResult

_status_message = {
    'success': 'Optimization terminated successfully.',
    'maxiter': 'Maximum number of iterations has been exceeded.',
    'conviol': 'Unable to satisfy constraints at end of iterations.',
    'pr_loss': 'Desired error not necessarily achieved due to precision loss.'
}


def uconmin_pso(fun, x0, friction=.8, max_velocity=1., g_rate=.1, l_rate=.1,
                max_iter=1e5, stable_iter=1e4, ptol=1e-5, callback=None):
    """Unconstrained minimization of the objective function using PSO.

    Parameters
    ----------
    For the documentation on the parameters, see ``pso.minimize_pso``.

    Returns
    -------
    result : scipy.optimize.OptimizeResult
        The optimization result represented as a OptimizeResult object. The
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
    .. Explain the algorithm here.

    References
    ----------
    .. Add citations here.

    Examples
    --------
    .. Use this function directly.
    """
    # Initial position and velocity.
    position = x0
    velocity = np.random.uniform(-max_velocity, max_velocity, position.shape)

    # Initial local best for each point and global best.
    pbest = position.copy()
    gbest = pbest[np.argmax(fun(pbest))]

    stable_max = 0
    stable_count = 0
    for ii in range(max_iter):
        # Store old for threshold comparison.
        gbest_fit = fun(gbest)

        # Determine the velocity gradients.
        dv_g = g_rate * np.random.uniform(0, 1) * (gbest - position)
        dv_l = l_rate * np.random.uniform(0, 1) * (pbest - position)

        # Update velocity such that |velocity| <= max_velocity.
        velocity *= friction
        velocity += (dv_g + dv_l)
        chk = (np.abs(velocity) > max_velocity)
        velocity[chk] = np.sign(velocity[chk]) * max_velocity

        # Update the local and global bests.
        position += velocity
        to_update = (np.apply_along_axis(fun, 1, position) < fun(pbest))

        if to_update.any():
            pbest[to_update] = position[to_update]
            gbest = pbest[np.argmax(fun(pbest))]

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

    if ii == max_iter:
        status = 1
        message = _status_message['maxiter']
    else:
        status = 0
        message = _status_message['success']

    return OptimizeResult(
        x=gbest, success=(stable_count == stable_iter), status=status,
        message=message, nit=ii, nsit=stable_max, fun=fun(gbest))


def conmin_pso(fun, x0, confunc, friction=.8, max_velocity=1., g_rate=.1,
               l_rate=.1, max_iter=1e5, stable_iter=1e4, ptol=1e-5, ctol=1e-5,
               callback=None):
    """Unconstrained minimization of the objective function using PSO.

    Parameters
    ----------
    confunc : callable
        The function constructed by check_constraints. Should return the
        constraint matrix.

    For the documentation on the rest of the parameters, see
    ``pso.minimize_pso``.

    Returns
    -------
    result : scipy.optimize.OptimizeResult
        The optimization result represented as a OptimizeResult object. The
        following additional keys are supported for the constrained case,

            cvec : float
                The constraint vector for x.

        For the documentation on the rest of the keys, see ``pso.uconmin_pso``.

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
    position = x0
    velocity = np.random.uniform(-max_velocity, max_velocity, position.shape)

    # Initial local best for each point and global best.
    lbest = position.copy()
    gbest = lbest[np.argmax(fun(lbest))]

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
            gbest = lbest[np.argmax(fun(lbest))]

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

    return OptimizeResult(
        x=gbest, success=(stable_count == stable_iter), status=status,
        message=message, nit=ii, nsit=stable_max, fun=fun(gbest),
        cvec=confunc(gbest[None]))
