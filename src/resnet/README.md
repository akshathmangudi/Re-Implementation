What is ResNet? 

The CNN is a fundamental architecture that's still used heavily today in Machine Learning, 
however it has a few drawbacks. As the number of layers, the gradients of the model converge
to zero or diverge to an infinitely large amount, known as the Vanishing/Exploding Gradient. 

This could be solved by reducing the number of layers, but the error rate increases if so. In order
to solve the vanishing gradient problem, the new architecture known as ResNet introduced 
Residual Blocks. 

A residual block, skips a few layers in between and connects the activation of later layers. 

The main advantage is that by training very large convolutional neural networks, for classification of 
hundreds of classes, the problem of the vanishing/exploding gradient is solved with the help of these
"skip connections". 

You can refer for more information here: [Introduction to Residual Networks](https://www.mygreatlearning.com/blog/resnet/)
