"""PSOPy: Python implementation for Particle Swarm Optimization

PSOPy (pronounced "Soapy") is a SciPy compatible super fast Python
implementation for Particle Swarm Optimization.

"""

from __future__ import absolute_import
from __future__ import print_function

import re
from glob import glob
from os.path import basename
from os.path import dirname
from os.path import join
from os.path import splitext

from setuptools import find_packages
from setuptools import setup


def read(*names, **kwargs):
    return open(
        join(dirname(__file__), *names),
        encoding=kwargs.get('encoding', 'utf-8')
    ).read()


setup(
    name='psopy',
    version='0.2.2',
    description='A SciPy compatible super fast Python implementation for '
                'Particle Swarm Optimization. ',
    long_description='%s' % (
        re.compile('^.. start-badges.*^.. end-badges',
                   re.M | re.S).sub('', read('README.rst'))
    ),
    author='Abhijit Theophilus, Dr. Snehanshu Saha, Suryoday Basak',
    author_email='abhijit.theo@gmail.com, snehanshusaha@gmail.com, '
                 'suryodaybasak@gmail.com',
    url='https://github.com/jerrytheo/psopy',
    packages=find_packages(),
    py_modules=[splitext(basename(path))[0] for path in glob('psopy/*.py')],
    include_package_data=True,
    zip_safe=False,

    # Classifiers list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Mathematics',
        'License :: OSI Approved :: BSD License',

        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Operating System :: Unix',

        # Supported Python versions.
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    keywords=[
        'optimization-algorithms',
        'particle-swarm-optimization',
        'constrained-optimization',
    ],
    install_requires=[
        'scipy',
        'numpy'
    ],
    setup_requires=[
        'pytest-runner'
    ],
    tests_require=[
        'pytest'
    ],
)
