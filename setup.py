from setuptools import setup, find_packages

setup(
    name='wcs',
    version='0.0.6',
    description='wcs tools for colab and co',
    url='git@github.com/snowdd1/wcs.git',
    author='Raoul',
    author_email='snowdd1.theelk@xoxy.net',
    license='MIT2.0',
    packages=find_packages(), # Auto setup all packages
    zip_safe=False
)