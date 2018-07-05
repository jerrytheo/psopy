import numpy as np
from psopy import minimize
from psopy import init_feasible


class TestConstrained:

    """Test for verifying constrained optimization."""

    def test_rosendisc(self):
        """Test against the Rosenbrock function constrained to a disk."""

        cons = ({'type': 'ineq', 'fun': lambda x: -x[0]**2 - x[1]**2 + 2},)
        x0 = init_feasible(cons, low=-1.5, high=1.5, shape=(1000, 2))
        sol = np.array([1., 1.])

        def rosen(x):
            return (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2

        res = minimize(rosen, x0)
        converged = res.success

        assert converged, res.message
        np.testing.assert_array_almost_equal(sol, res.x, 3)

    def test_mishra(self):
        """Test against the Mishra Bird function."""

        cons = (
            {'type': 'ineq', 'fun': lambda x: 25 - np.sum((x + 5) ** 2)},)
        x0 = init_feasible(cons, low=-10, high=0, shape=(1000, 2))
        sol = np.array([-3.130, -1.582])

        def mishra(x):
            cos = np.cos(x[0])
            sin = np.sin(x[1])
            return sin*np.e**((1 - cos)**2) + cos*np.e**((1 - sin)**2) + \
                (x[0] - x[1])**2

        res = minimize(mishra, x0)
        converged = res.success

        assert converged, res.message
        np.testing.assert_array_almost_equal(sol, res.x, 3)
