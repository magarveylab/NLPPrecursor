from setuptools import setup

with open("requirements.txt") as fp:
    required = fp.read().splitlines()

setup(
    name='nlpprecursor',
    version='0.0.1',
    install_requires=required,
    packages=['nlpprecursor', 'nlpprecursor.classification', 'nlpprecursor.annotation',
        'nlpprecursor.annotation.models', 'nlpprecursor.annotation.tests',
        'nlpprecursor.classification.tests'],
    url='',
    license='',
    author='nmerwin',
    author_email='',
    description='',
)
