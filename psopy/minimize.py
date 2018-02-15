import functools as ft
import numpy as np

from pso import _minimize_pso
from constraints import gen_confunc


def minimize_pso(fun, x0, args=(), constraints=(), tol=None, callback=None,
                 options=None):
    """Minimization of scalar function of one or more variables using Particle
    Swarm Optimization (PSO).

    Parameters
    ----------
    fun : callable
        The objective function to be minimized. Must be in the form
        ``fun(x, *args)``. The optimizing argument, ``x``, is a 1-D array of
        points, and ``args`` is a tuple of any additional fixed parameters
        needed to completely specify the function.
    x0 : array_like of shape (N, D)
        Initial position to begin PSO from, where ``N`` is the number of points
        and ``D`` the dimensionality of each point.
    args : tuple, optional
        Extra arguments passed to the objective function.
    constraints : tuple, optional
        Constraints definition. Each constraint is defined in a dictionary with
        fields:

            type : str
                Constraint type: ‘eq’ for equality, ‘ineq’ for inequality.
            fun : callable
                The function defining the constraint.
            args : sequence, optional
                Extra arguments to be passed to the function.

        Equality constraint means that the constraint function result is to be
        zero whereas inequality means that it is to be non-negative.
    tol : float, optional
        Tolerance for termination and constraint adjustment. For detailed
        control, use options.
    callback : callable, optional
        Called after each iteration on each solution vector.
    options : dict, optional
        A dictionary of solver options:

            friction: float, optional
                Velocity is scaled by friction before updating, default 0.8.
            g_rate: float, optional
                Global learning rate, default 0.1.
            l_rate: float, optional
                Local (or particle) learning rate, default 0.1.
            max_velocity: float, optional
                Threshold for velocity, default 1.0.
            max_iter: int, optional
                Maximum iterations to wait for convergence, default 10000.
            stable_iter: int, optional
                Number of iterations to wait before Swarm is declared stable,
                default 1000.
            ptol: float, optional
                Change in position should be greater than ``ptol`` to update,
                otherwise particle is considered stable, default 1e-5.
            ctol: float, optional
                Acceptable error in constraint satisfaction, default 1e-5.
            sttol : float, optional
                Tolerance to convert strict inequalities to non-strict
                inequalities, default 1e-6.
            eqtol : float, optional
                Tolerance to convert equalities to non-strict inequalities,
                default 1e-7.

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
            cvec : float
                The constraint vector for x. (only when constraints specified)

    See Also
    --------
    `psopy._minimize_pso` : The internal implementation for PSO used by this
        function. May be a little faster to use directly.

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

    ** Constrained Optimization **

    The standard PSO algorithm does not guarantee that the individual solutions
    will converge to a feasible global solution. To solve this, each particle
    selects another particle, called the leader and uses this particle's
    ``pbest`` value instead of its own to update its velocity. The leader for
    a given particle is selected by picking the particle whose ``pbest`` is
    closest to the current position of the given particle. Further, ``pbest``
    is updated only those particles where the sum of the constraint vector is
    less than the constraint tolerance.

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
    .. [3] Cite our paper and add reference.

    Examples
    --------
    Let us consider the problem of minimizing the Rosenbrock function,
    implemented as `scipy.optimize.rosen`.

    >>> import numpy as np
    >>> from scipy.optimize import rosen

    Initialize 1000 particles and run the minimization function:

    >>> x0 = np.random.uniform(0, 2, (1000, 5))
    >>> res = minimize_pso(fun, x0, options={'stable_iter': 50})
    >>> res.x
    array([ 1.,  1.,  1.,  1.,  1.])

    Next, we consider a minimization problem with several constraints. The
    objective function is:

    >>> fun = lambda x: (x[0] - 1)**2 + (x[1] - 2.5)**2

    The constraints defined as:

    >>> cons = ({'type': 'ineq', 'fun': lambda x:  x[0] - 2 * x[1] + 2},
    ...         {'type': 'ineq', 'fun': lambda x: -x[0] - 2 * x[1] + 6},
    ...         {'type': 'ineq', 'fun': lambda x: -x[0] + 2 * x[1] + 2},
    ...         {'type': 'ineq', 'fun': lambda x: -x[0]},
    ...         {'type': 'ineq', 'fun': lambda x: -x[1]})
    cons = ({'type': 'ineq', 'fun': lambda x:  x[0] - 2 * x[1] + 2},
            {'type': 'ineq', 'fun': lambda x: -x[0] - 2 * x[1] + 6},
            {'type': 'ineq', 'fun': lambda x: -x[0] + 2 * x[1] + 2},
            {'type': 'ineq', 'fun': lambda x: -x[0]},
            {'type': 'ineq', 'fun': lambda x: -x[1]})

    Running the constrained version:

    >>> minimize_pso(fun, x0, constrainsts=cons)

    """
    x0 = np.asarray(x0)
    if x0.dtype.kind in np.typecodes["AllInteger"]:
        x0 = np.asarray(x0, dtype=float)

    if not isinstance(args, tuple):
        args = (args,)

    if not options:
        options = {}

    conargs = {}
    if tol is not None:
        options.setdefault('ptol', tol)
        options.setdefault('ctol', tol)
        conargs['sttol'] = options.pop('sttol', tol)
        conargs['eqtol'] = options.pop('eqtol', tol)

    fun_ = ft.update_wrapper(
        lambda x: np.apply_along_axis(fun, 1, x, *args), fun)

    if callback is not None:
        options['callback'] = ft.update_wrapper(
            lambda x: np.apply_along_axis(callback, 1, x), callback)

    if constraints:
        options['confunc'] = gen_confunc(constraints, **conargs)

    return _minimize_pso(fun_, x0, **options)
