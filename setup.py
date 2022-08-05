from setuptools import setup, find_packages
from os import path


setup(name = 'PyDataDrivenReachabilityAnalysis',
    packages=find_packages(),
    version = '0.0.1',
    description = 'Python library for Data-Driven Reachability Analysis',
    url = 'https://github.com/rssalessio/PyDeePC',
    author = 'Alessio Russo',
    author_email = 'alessior@kth.se',
    install_requires=['numpy', 'scipy'],
    license='MIT',
    zip_safe=False,
    python_requires='>=3.7',
)