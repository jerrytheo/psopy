import numpy as np
from psopy import minimize_pso, gen_confunc
from scipy.optimize import rosen


class TestQuick:

    def test_unconstrained(self):
        x0 = np.random.uniform(0, 2, (1000, 5))
        sol = np.array([1., 1., 1., 1., 1.])
        res = minimize_pso(rosen, x0)
        converged = res.success
        assert converged, res.message
        np.testing.assert_array_almost_equal(sol, res.x, 3)

    def test_constrained(self):
        x0 = np.random.uniform(0, 2, (1000, 2))
        cons = ({'type': 'ineq', 'fun': lambda x:  x[0] - 2 * x[1] + 2},
                {'type': 'ineq', 'fun': lambda x: -x[0] - 2 * x[1] + 6},
                {'type': 'ineq', 'fun': lambda x: -x[0] + 2 * x[1] + 2},
                {'type': 'ineq', 'fun': lambda x: x[0]},
                {'type': 'ineq', 'fun': lambda x: x[1]})
        cfunc = gen_confunc(cons)
        condn = (cfunc(x0).sum(1) != 0)
        while condn.any():
            x0[condn] = np.random.uniform(0, 2, x0[condn].shape)
            condn = (cfunc(x0).sum(1) != 0)
        options = {'g_rate': 1., 'l_rate': 1., 'max_velocity': 4.,
                   'stable_iter': 50}

        sol = np.array([1.4, 1.7])
        res = minimize_pso(lambda x: (x[0] - 1)**2 + (x[1] - 2.5)**2, x0,
                           constraints=cons, options=options)
        converged = res.success
        assert converged, res.message
        np.testing.assert_array_almost_equal(sol, res.x, 3)
