# AutoGrad: A Learning Tool for Automatic Differentiation. Become a true backprop ninja!

![AutoGrad Illustration](./images/autograd.gif)


This project features an autograd engine that follow patterns from Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd). It has two abstractions:

1. A minimal scalar-valued class that provides a clear understanding of basic neural network operations.(Start here!)
2. An optimized tensor class that demonstrates improved performance techniques.

Both classes implement backpropagation through dynamically constructed computational graphs (DAGs). The project also includes neural network implementations built on top of both classes. I made this for students and enthusiasts looking to grasp the fundamentals of neural networks and automatic differentiation! Hope this helps!
- Install with pip install -e . The only runtime dependency is numpy.
