"""
utilities.py - Utilities for Printing and Saving
=================================================

Utility functions to enable verbose status messages and saving convergence data
in a CSV file if required.

=================== ===========================================================
Functions
===============================================================================
setup_print         Sets up the print options and prints the header.
=================== ===========================================================

"""

import numpy as np


def setup_print(xlen, max_iter, constraints=False):
    """Set up print options and print the header.

    Parameters
    ----------
    ashape : int
        Number of dimensions of a point.
    max_iter : int
        Maximum number of total iterations.
    constraints : bool
        Whether constrained or unconstrained optimization being performed.

    Returns
    -------
    message : str
        Format specifier for convergence messages.

    """
    np.set_printoptions(
        precision=4, threshold=4, edgeitems=2, suppress=True, sign=' ',
        floatmode='fixed')

    xcol_len = 1 + 8 * xlen if xlen < 5 else 37
    itercol_len = max(len(str(max_iter)), 4)

    cvcol_head = '  {:>7.5}' if constraints else ''
    cvcol_body = '  {:>7.5f}' if constraints else ''
    itercol = '{:>' + str(itercol_len) + '}'
    xcol = '{:>' + str(xcol_len) + '}'

    headmsg = itercol + '  ' + xcol + '  {:>7.4}' + cvcol_head
    message = itercol + '  {}  {:>7.4f}' + cvcol_body

    header = headmsg.format('Iter', 'GBest', 'Func', 'MaxCV')
    print(header)
    print('-' * len(header))

    return message
