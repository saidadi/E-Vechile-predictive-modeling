from setuptools import setup, find_packages

setup(
    name='EVehiclepopulation_Analysis',
    version='0.1.0',
    description='analysis',
    author='Sai krishna Dadi',
    author_email='sdadi@mail.yu.edu',
    license='MIT',
    packages=['modeling_packages'],
    install_requires=[
        'matplotlib>=3.0.2',
        'numpy>=1.15.2',
        'pandas>=0.23.4',
        'seaborn>=0.11.0',
        'scikit-learn>=1.3.2'
    ],
)
