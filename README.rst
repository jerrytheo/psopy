===============================================================================
PSOPy
===============================================================================

    A python implementation of Particle Swarm Optimization.

PSOPy (pronounced "Soapy") is a SciPy compatible super fast Python
implementation for Particle Swarm Optimization. The codes are tested for
standard optimization test functions (both constrained and unconstrained).

The library provides two implementations, one that mimics the interface to
`scipy.optimize.minimize` and one that directly runs PSO. The SciPy compatible
function is a wrapper over the direct implementation, and therefor may be
slower in execution time, as the constraint and fitness functions are wrapped.

-------------------------------------------------------------------------------
Authors
-------------------------------------------------------------------------------

- Abhijit Jeremiel Theophilus, abhijit.theo@gmail.com
- Dr\. Snehanshu Saha, snehanshusaha@pes.edu
- Suryoday Basak, suryodaybasak@gmail.com

-------------------------------------------------------------------------------
License
-------------------------------------------------------------------------------

| Licensed under the BSD 3-Clause License.
| Copyright 2018 Abhijit Theophilus, Snehanshu Saha, Suryoday Basak

-------------------------------------------------------------------------------
References
-------------------------------------------------------------------------------
1. Ray, T. and Liew, K.M., 2001. A swarm with an effective information
   sharing mechanism for unconstrained and constrained single objective
   optimisation problems. In Evolutionary Computation, 2001. Proceedings
   of the 2001 Congress on (Vol. 1, pp. 75-80). IEEE.
2. Eberhart, R. and Kennedy, J., 1995, October. A new optimizer using
   particle swarm theory. In Micro Machine and Human Science, 1995.
   MHS'95., Proceedings of the Sixth International Symposium on (pp.
   39-43). IEEE.
3. Shi, Y. and Eberhart, R., 1998, May. A modified particle swarm
   optimizer. In Evolutionary Computation Proceedings, 1998. IEEE World
   Congress on Computational Intelligence., The 1998 IEEE International
   Conference on (pp. 69-73). IEEE.
4. Cite our paper and add reference.
