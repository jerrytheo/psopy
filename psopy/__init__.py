"""
===============================================================================
PSOpy : Python Implementation for Particle Swarm Optimization.
===============================================================================

PSOPy (pronounced "Soapy") is a SciPy compatible super fast Python
implementation for Particle Swarm Optimization. The codes are tested for
standard optimization test functions (both constrained and unconstrained).

The library provides two implementations, one that mimics the interface to
``scipy.optimize.minimize`` and one that directly runs PSO. The SciPy
compatible function is a wrapper over the direct implementation, and therefor
may be slower in execution time, as the constraint and fitness functions are
wrapped.

The implementation for unconstrained optimization follows that laid out by
Eberhart and Kennedy [1]_ with the adjustment for inertia discussed by Shi
and Eberhart [2]_. The paper by Lorem and Ipsum [3]_ discusses a variation
to the standard PSO algorithm that allows the search space to be constrained.


=================== ==================================================
Functions
======================================================================
gen_confunc         Generate the constraint function.
init_feasible       Initialize feasible solutions.
minimize_pso        SciPy compatible interface to ``pswarmopt``.
pswarmopt           Optimize under constraints using a particle swarm.
=================== ==================================================

-------------------------------------------------------------------------------
References
-------------------------------------------------------------------------------

.. [1] Eberhart, R. and Kennedy, J., 1995, October. A new optimizer using
    particle swarm theory. In Micro Machine and Human Science, 1995. MHS'95.,
    Proceedings of the Sixth International Symposium on (pp. 39-43). IEEE.
.. [2] Shi, Y. and Eberhart, R., 1998, May. A modified particle swarm
    optimizer. In Evolutionary Computation Proceedings, 1998. IEEE World
    Congress on Computational Intelligence., The 1998 IEEE International
    Conference on (pp. 69-73). IEEE.
.. [3] Lorem ipsum dolor sit amet, consectetur adipiscing elit. Integer
    volutpat feugiat imperdiet. Phasellus placerat elit nec erat tristique
    faucibus. Suspendisse at nunc odio. Nullam sagittis nunc ut sed.

"""


from .constraints import gen_confunc, init_feasible
from .minimize import minimize, _minimize_pso

__all__ = [
    'gen_confunc',
    'init_feasible',
    'minimize',
    '_minimize_pso'
]
