#! -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name='bert4keras',
    version='0.11.3',
    description='an elegant bert4keras',
    long_description='bert4keras: https://github.com/bojone/bert4keras',
    license='Apache License 2.0',
    url='https://github.com/bojone/bert4keras',
    author='bojone',
    author_email='bojone@spaces.ac.cn',
    install_requires=['keras<=2.3.1'],
    packages=find_packages()
)
