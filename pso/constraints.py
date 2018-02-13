import numpy as np


def gen_confunc(constraints, sttol=1e-6, eqtol=1e-7):
    """Convert the list of constraints to a function that returns the
    constraint matrix when run on the position matrix.

    Strict inequalities of the form ``g(x) <= 0`` are converted to non-strict
    inequalities ``g(x) + sttol <= 0``.

    Equality constraints of the form ``g(x) = 0`` are converted to a pair of
    inequality constraints::

         g(x) - eqtol <= 0
        -g(x) - eqtol <= 0

    Parameters
    ----------
    constraints : tuple
        Constraints definition. Each constraint is defined in a dictionary with
        fields:

            type : str
                Constraint type: ‘eq’ for equality, 'stin' for strict
                inequality, ‘ineq’ for inequality.
            fun : callable
                The function defining the constraint.
            args : sequence, optional
                Extra arguments to be passed to the function.

        Equality constraint means that the constraint function result is to be
        zero, strict inequality means that it is to be negative and inequality
        means that it is to be non-negative.
    sttol : float, optional
        Tolerance to convert strict inequalities to non-strict inequalities.
    eqtol : float, optional
        Tolerance to convert equalities to non-strict inequalities.

    Returns
    -------
    check_constraints : callable
        When called with the particle positions, `check_constraints(x)` returns
        the constraint matrix where the element at `(i,j)` indicates the extent
        to which solution `i` violates constraint `j`.

    Notes
    -----
    Ray and Liew [1]_ describe a representation for nonstrict inequality
    constraints when optimizing using a particle swarm. However, strict
    inequality and equality constraints need to be converted to non-strict
    inequalities. Introducing the tolerance `sttol` converts strict
    inequality constraints and `eqtol` converts the equality constraints by
    wrapping over the corresponding function ``fun``. Thus, if the origial
    problem contained ``q`` inequality and ``r`` equality constraints, we now
    have ``s = q + 2r`` constraints specified by these wrapped functions.

    The returned function `check_constraints` takes the position matrix and
    returns the constraint matrix where the element at ``(i,j)`` is given by::

        C_ij = max(g'_j(x_i), 0)

    where, ``g'_j`` is the wrapped function for constraint ``j``, ``x_i`` is
    the ``i``th position vector.

    This function is primarily for use within `pso.minimize_pso` to convert
    SciPy style constraints to the technique used in this implementation. There
    may be some overhead during execution due to the recursive function call
    used to implement the conversion.

    References
    ----------
    .. [1] Ray, T. and Liew, K.M., 2001. A swarm with an effective information
        sharing mechanism for unconstrained and constrained single objective
        optimisation problems. In Evolutionary Computation, 2001. Proceedings
        of the 2001 Congress on (Vol. 1, pp. 75-80). IEEE.

    Examples
    --------
    Consider the constraints::

        0 <= x < 1
        0 <= y < 1
        x + y = 1

    These constraints are converted to the form required for the problem
    using,

    >>> constraints = (
    ...     {'type': 'ineq', 'fun': lambda x: -x[0]},
    ...     {'type': 'ineq', 'fun': lambda x: -x[1]},
    ...     {'type': 'stin', 'fun': lambda x: x[0] - 1},
    ...     {'type': 'stin', 'fun': lambda x: x[0] - 1},
    ...     {'type': 'eq', 'fun': lambda x: x[0] + x[1]}
    ... )
    >>> confunc = gen_confunc(constraints, sttol=0.001, eqtol=0.0001)

    To test the function, generate a test position matrix and run the function
    on this test matrix.

    >>> test_pos = np.array([
    ...     [ 0.3,  0.7],
    ...     [-0.4,  1.4],
    ...     [ 0.4,  0.4],
    ... ])
    >>> confunc(test_pos)
    array([[ 0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ],
           [ 0.4   ,  0.    ,  0.    ,  0.401 ,  0.    ,  0.    ],
           [ 0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.1999]])

    """

    funlist = []

    # For each kind of constraint, the resulting adjusted constraint is placed
    # in the list. Equality constraints are split into two constraints.
    for con in constraints:
        fun = con['fun']
        args = con.get('args', [])

        if con['type'] == 'eq':
            funlist.extend((
                lambda x, fun=fun, args=args: np.max((
                    -eqtol + fun(x, *args), 0)),
                lambda x, fun=fun, args=args: np.max((
                    -eqtol - fun(x, *args), 0)),
            ))

        elif con['type'] == 'ineq':
            funlist.append(
                lambda x, fun=fun, args=args: np.max((fun(x, *args), 0))
            )

        elif con['type'] == 'stin':
            funlist.append(
                lambda x, fun=fun, args=args: np.max((
                    sttol + fun(x, *args), 0))
            )

    def _check(x):
        """Run the functions defined in funlist for x."""
        for f in funlist:
            print(f(x))
        return np.array([f(x) for f in funlist])

    def check_constraints(points):
        """Returns the constraint matrix for the given set of points.

        Each constraint is of the form ``g(x) <= 0``. The element at ``(i,j)``,
        of the matrix indicates the extent to which solution ``i`` violates
        constraint ``j``, i.e., ``C_ij = g_j(x_i)``.

        Parameters
        ----------
        points : array_like of shape (N, D)
            The current positions of each particle in the swarm, where ``N`` is
            the number of particles and ``D`` is the number of dimensions.

        Returns
        -------
        conmatrix : ndarray of shape (N, S)
            The constraint matrix, where ``N`` is the number of particles and
            ``S`` is the number of constraints after adjusting with a
            tolerance.
        """
        return np.apply_along_axis(_check, axis=1, arr=points)

    return check_constraints
