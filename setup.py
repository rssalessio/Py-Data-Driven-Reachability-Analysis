from setuptools import setup, find_packages
from os import path


setup(name = 'pydatadrivenreachability',
    packages=find_packages(),
    version = '0.0.2',
    description = 'Python library for Data-Driven Reachability Analysis',
    url = 'https://github.com/rssalessio/Py-Data-Driven-Reachability-Analysis',
    author = 'Alessio Russo',
    author_email = 'alessior@kth.se',
    install_requires=['numpy', 'scipy', 'cvxpy'],
    license='MIT',
    zip_safe=False,
    python_requires='>=3.7',
)