import functools as ft
import numpy as np
from .minimize import uconmin_pso, conmin_pso
from .utils import gen_confunc


def minimize_pso(fun, x0, args=(), constraints=(), tol=None, callback=None,
                 options=None):
    """Minimize the scalar function through particle swarm optimization.

    Parameters
    ----------
    fun : callable
        The objective function to be minimized. Must be in the form
        ``f(x, *args)``. The optimizing argument, ``x``, is a 1-D array of
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
    res : scipy.optimize.OptimizeResult
        The optimization result represented as a OptimizeResult object.

    ==> need to add notes, references, examples
        --> Explain method used for constrained.
        --> Explain method used for unconstrained.
        --> Explain how to execute the faster model.
    """
    x0 = np.asarray(x0)
    if x0.dtype.kind in np.typecodes["AllInteger"]:
        x0 = np.asarray(x0, dtype=float)

    if tol is not None:
        options.setdefault('ptol', tol)
        options.setdefault('ctol', tol)

        sttol = options.pop('sttol', tol)
        eqtol = options.pop('eqtol', tol)

    fun_ = ft.update_wrapper(lambda x: fun(x, *args), fun)

    if constraints:
        confunc = gen_confunc(constraints, sttol, eqtol)
        result = conmin_pso(fun_, x0, confunc, **options)
    else:
        result = uconmin_pso(fun_, x0, **options)

    return result
