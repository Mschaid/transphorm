#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [ ]

test_requirements = ['pytest>=3', ]

setup(
    author="Mike Schaid",
    author_email='michael.schaid@northwestern.edu',
    python_requires='>=3.10',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
    ],
    description="source code for lerner lab database managment system",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='transphorm',
    name='transphorm',
    packages=find_packages(include=['transphorm', 'transphorm.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/Mschaid/transphorm',
    version='0.1.0',
    zip_safe=False,
)
