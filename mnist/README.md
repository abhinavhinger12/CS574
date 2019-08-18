# CS574
 Assignments for Computer Vision and Machine Learning course

### Setting up the Environment
Create a new conda environment and run. ```conda install -c pytorch -c fastai fastai```. This will install the pytorch build with the latest cudatoolkit version and all other relevant dependencies. Code is in form of Jupyter Notebook because of higher modularity and readability.

### MNIST 
#### CNN: 
We'll use two 2-D convolutional layers followed by two fully-connected (or linear) layers. As
activation function we'll choose rectified linear units (ReLUs in short) and as a means of
regularization we'll use two dropout layers.
    