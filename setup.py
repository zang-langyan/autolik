# python3 -m pip install -e .
from setuptools import setup

setup(
    name='autolik',
    version='0.1',
    description='Provide the automatic differentiation for Likelihood maximization routine',
    author='Langyan Zang',
    author_email="langyan.zang@uzh.ch",
    packages=['autolik', 'autolik.Dual', 'autolik.distributions', 'autolik.likelihood', 'autolik.optim'],
    install_requires=['scipy'],
    keywords=['Automatic differentiation', 'gradients',
              'likelihood', 'optimization','Python', 
              'Scipy'],
    url='https://github.com/7lang2yan/autolik.git',
    license='MIT',
    classifiers=['Development Status :: 4 - Beta',
                 'License :: OSI Approved :: MIT License',
                 'Programming Language :: Python :: 3.9'],
)