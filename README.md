# AutoGrad: A Learning Tool for Automatic Differentiation 

![AutoGrad Illustration](./images/autograd.gif)

This project features two autograd engines that follow patterns from Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd):

1. A minimal scalar-valued engine that provides a clear understanding of basic neural network operations.
2. An optimized tensor engine that demonstrates improved performance techniques.

Both engines implement backpropagation through dynamically constructed computational graphs (DAGs), offering insight into the core principles of automatic differentiation. The project also includes compact neural network libraries built on top of both engines, with an API similar to PyTorch. While not as efficient as industry-standard libraries, this system serves as an excellent learning tool, capable of creating simple deep neural networks. Ideal for students and enthusiasts looking to grasp the fundamentals of neural networks and automatic differentiation.
