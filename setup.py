from setuptools import setup, find_packages

setup(
    name='modeling_packages',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'matplotlib',
        'scikit-learn',
    ],
)
