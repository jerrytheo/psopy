import numpy as np


def convert_constraints(constraints, sttol=1e-6, eqtol=1e-7):
    """Convert the list of constraints to a function that returns the
    constraint matrix when run on a solution vector.

    Strict inequalities of the form ``g(x) <= 0`` are converted to non-strict
    inequalities ``g(x) + sttol <= 0``.

    Each equality constraint of the form ``g(x) = 0`` is converted to
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
        When called with the partickle positions, ``check_constraints(x)``
        returns the constraint matrix where the element at ``(i,j)`` indicates
        the extent to which solution ``i`` violates constraint ``j``.

    < need to add notes, references, examples >
    """

    funlist = []

    # For each kind of constraint, the resulting adjusted constraint is placed
    # in the list. Equality constraints are split into two constraints.
    for con in constraints:
        if con['type'] == 'eq':
            funlist.append(
                lambda x: np.max(-eqtol + con['fun'](x, *con['args']), 0),
                lambda x: np.max(-eqtol - con['fun'](x, *con['args']), 0),
            )
        elif con['type'] == 'ineq':
            funlist.append(
                lambda x: np.max(sttol + con['fun'](x, *con['args']), 0)
            )
        elif con['type'] == 'stin':
            funlist.append(
                lambda x: np.max(con['fun'](x, *con['args']), 0)
            )

    def _check(x):
        """Run the functions defined in funlist for x."""
        np.array((f(x) for f in funlist))

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
