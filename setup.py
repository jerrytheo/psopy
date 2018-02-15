"""A setuptools based setup module.
See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='psopy',
    version='0.0.1a1',
    description='A SciPy compatible super fast Python implementation for '
                'Particle Swarm Optimization. ',
    long_description=long_description,
    # url='https://github.com/pypa/sampleproject',
    author='Abhijit Theophilus, Dr. Snehanshu Saha, Suryoday Basak',
    author_email='abhijit.theo@gmail.com, snehanshusaha@gmail.com, ' \
                 'suryodaybasak@gmail.com',

    # Classifiers list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Mathematics',

        # License used.
        'License :: OSI Approved :: BSD License',

        # To be done later ...(Need to add support for Python 2.)
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],

    keywords='particle-swarm-optimization constrained-optimization',
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),  # Required

    # https://packaging.python.org/en/latest/requirements.html
    install_requires=['scipy', 'numpy'],
    extras_require={
        'test': ['pytest'],         # $ pip install psopy[test]
    },

    # To be done later ...
    # If there are data files included in your packages that need to be
    # installed, specify them here.
    package_data={
        'psopy': [],
    },

    # Look up entry_points if scripting required.

)
