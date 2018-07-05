import numpy as np
from psopy import minimize


class TestUnconstrained:

    """Test for verifying unconstrained optimization."""

    def test_ackley(self):
        """Test against the Ackley function."""

        x0 = np.random.uniform(-5, 5, (1000, 2))
        sol = np.array([0., 0.])

        def ackley(x):
            return -20 * np.exp(-.2 * np.sqrt(0.5 * (x[0]**2 + x[1]**2))) - \
                np.exp(.5 * (np.cos(2*np.pi*x[0]) + np.cos(2*np.pi*x[1]))) + \
                np.e + 20

        res = minimize(ackley, x0)
        converged = res.success

        assert converged, res.message
        np.testing.assert_array_almost_equal(sol, res.x, 3)

    def test_levi(self):
        """Test against the Levi function."""

        x0 = np.random.uniform(-10, 10, (1000, 2))
        sol = np.array([1., 1.])

        def levi(x):
            sin3x = np.sin(3*np.pi*x[0]) ** 2
            sin2y = np.sin(2*np.pi*x[1]) ** 2
            return sin3x + (x[0] - 1)**2 * (1 + sin3x) + \
                (x[1] - 1)**2 * (1 + sin2y)

        res = minimize(levi, x0)
        converged = res.success

        assert converged, res.message
        np.testing.assert_array_almost_equal(sol, res.x, 3)
