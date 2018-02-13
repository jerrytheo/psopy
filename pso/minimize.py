import numpy as np
from scipy.spatial import distance


def _pso(fun, x0, confunc=None, friction=.8, g_rate=.1, l_rate=.1,
         max_velocity=1., max_iter=1e5, stable_iter=1e4, ptol=1e-5,
         ctol=1e-5, callback=None):
    """Internal function to minimize the objective function through PSO.

    See Also
    --------
    For the rest of the documentation, see ``pso.minimize_pso``.

    Parameters
    ----------
    confunc : callable
        The function constructed by check_constraints. Should return the
        constraint matrix.
    """
    # Initial position and velocity.
    position = x0
    velocity = np.random.uniform(-max_velocity, max_velocity, position.shape)

    # Initial local best for each point and global best.
    lbest = position.copy()
    gbest = lbest[np.argmax(fun(lbest))]

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
                return (gbest, ii)
        else:
            stable_count = 0

        # Final callback.
        if callback is not None:
            position = np.apply_along_axis(callback, 1, position)
