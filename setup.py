from setuptools import setup
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='autograd',  
    version='0.1',  
    description="A small autodiff and neural network library patterned on Karpathy's micrograd,"
                "but with support for tensors expressed as numpy arrays.",
    author='kabatyy',
    long_description=long_description,
    author_email='sam.kabati@gmail.com',
    url='https://github.com/kabatyy/AutoGrad', 
    python_requires='>=3.12',
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy',
    ],
)

