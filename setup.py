#!/usr/bin/env python
from setuptools import setup, find_packages
import os

with open('requirements.txt') as fd:
    required = fd.read().splitlines()

setup(
    name='DataScienceCodingChallenge',
    version='1.0',
    packages=['codetemplate', 'codetemplate.src', 'codetemplate.operation', 'codetemplate.operation.tests',
              'codetemplate.operation.tests.data_validation'],
    url='',
    license='',
    author='Mitodru Niyogi',
    author_email='mitodru.niyogi@gmail.com',
    description='Zeiss DS Coding Challenge'
)
