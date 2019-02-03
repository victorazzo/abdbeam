#!/usr/bin/env python
from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    readme = fh.read()

setup(
    name='abdbeam',
    version='0.2.0',
    description='Cross section analysis of composite material beams of any shape.',
    long_description=readme,
    long_description_content_type='text/markdown'
    author='Danilo S. Victorazzo',
    author_email='victorazzo@gmail.com',
    url='https://github.com/victorazzo/abdbeam',
    license='BSD-3',
    packages=find_packages(exclude=('tests', 'docs')),
    classifiers=[
	    'Natural Language :: English',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3.6',
		'Programming Language :: Python :: 3.7',
        "License :: OSI Approved :: BSD License",
        "Topic :: Scientific/Engineering",
        "Operating System :: OS Independent",
    ],
)
