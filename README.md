# Project in the course: Advanced Machine Learning for Data Science

## I’m Something of a Painter Myself
**Marie-Louise Tommerup & Mia Pugholm**

General goal: *To what extent is it possible to identify and create pictures that resemble Monet painting-style using DCGan (Deep Convolutional generative adversarial network)?*

This project is inspired by a Kaggle competition - https://www.kaggle.com/competitions/gan-getting-started/overview 
The problem should be solved with a variant of a GAN model. We have decided to implement a DCGan to solve the problem and answer our own created research question. Creating a DCGan model we are working in the unsupervised machine learning paradigm. 

**The dataset:** 
- The data is provided in the Kaggle competition consists of ~300 Monet paintings and ~7000 photos, both in jpg and TFRecord format. As we are making use of a DCgan, only the Monet pictures were relevant to use in our case.

**Goals:**
- Generate 7000 pictures in the style of Monet
- In the early stages of the DCGAN process, we have a Discriminator goal as defined: the discriminator is able to identify a fake Monet and a real Monet. 
- In the later stages of the DCGAN process, we have a generator goal defined as Generating a picture that passes the discriminator as a real Monet painting.
- How does convolutional neural network contribute to the architecture of a GAN model.

### The DCGAN architecture
*“A Generative Adversarial Network (GAN) is a deep learning architecture that consists of two neural networks competing against each other in a zero-sum game framework. The goal of GANs is to generate new, synthetic data that resembles some known data distribution.”* Quote taken from https://www.geeksforgeeks.org/generative-adversarial-network-gan/

There are many variants of GAN models, since the genetator and discriminator is not set to be a specific machine learning model. Therefore for our problem, we have chosen to look into GAN variant that is called DCGAN, that include the CNN architecture in the generator and discriminator. The reason for this specific type of GAN model was chosen because there weren't many other programmers who had made a DCGAN in the Kaggle competition. 

Our implementation of the DCGAN is common, and is inspired by Pytorch own example: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

Choosing the hyperparameters for our model, we have used the same values as Radford et. al does in the paper: *Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks* (https://arxiv.org/pdf/1511.06434.pdf). Where they have concluded the best hyperparameters when working with DCGAN. For this reason we have used the same values for our hyperparamteres and has therefore instead experimented with other parts of the DCGAN, such as the picture size, number of epochs, batchsizes and depth in discriminator and generator. 

**The fixed hyperparameters are the following:** 
- Weights are initialized from a normal distribution with the standard deviation of 0.2. 
- Learning rate: 0.0002.
- Slope of LeakyReLU set to 0.2
- Momentum term beta1 is set to 0.5

**Training mechanisms:**
Setting up the model, we are initializing the weights of both the generator and the discriminator. During training, the generator network takes random noise as input and generates synthetic images, while the discriminator network tries to distinguish between real and synthetic images. The training process involves iteratively updating the weights of the generator and discriminator networks based on the error rate of the discriminator's classification of real and synthetic images. The Generator and Discriminator are evaluated by using the BCEloss (Binary Cross Entropy).

### Key experiments & results
*present and explain results, e.g. in simple accuracy tables over error graphs up to visualisations of representations and/or edge cases – keep it crisp

**Key experiments**
In general working with the DCGAN, we have tried to run 3 different types of DCGAN models, each of them create different size picture, 64x64, 128x128, 256x256. Where we in each model in the bigger pictures, needed to expand the model, and making it deeper in order to create the larger pictures. This could also be done by changing the size of the kernel for the generator and discriminator, though we choose to expand the model. 

The main model that we have run is the model that works with the picture sizes of 64x64. With the smaller size, it was easier to try out new parameters, as we found that working with the 128x128 and 256x256 pictures were very computational heavy. As mentioned earlier The hyperparameters that we have experimented with are the epochs and batchsizes in our model.

**Changes in epochs and batchsizes:**
- Epochs: 5, 50, 200, 500.
- Batch sizes: 10, 25

#### Results

Overall all our DCGAN models resulted in a very good discriminator and not as good generator, this is shown in our loss visualisations. 



### Discussion 
Overall we have learned how a DCGAN works, in the beginning of the project we had limited knowledge on how the generator and discriminator worked in the detail, though working with it and working with the Monet pictures, we have now gathered a deeper understanding, not only in the DCGAN but also in the general structure of a GAN model.  

**What is good**
- We were able to run all 3 different models with 200 epochs and a batchsize of 10. 
- Understand how the we can go from a vector of latens values to a picture with upscaling. 
- The output of the pictures actually looks somewhat like a painting with many different colors.  

**What can be improved**
- Run more the 3 different sizes of picture with the same hyper parameters to
- Further research what the amount of epoch and batch sizes that could give a good result. 
- Methods to solve the issue with the mode collapse (the pictures "getting stuck") 


### Github repository
This github repository contains 3 different notebooks, each containing 3 different DCGAN models. The models varies in the sizes of the pictures that they work with, they are producing the repective picture sizes: 64x64, 128x128, 256x256. 

In each of the notebooks there is various hyperparameters that can be modified, to change the outcome of the models.  

*To run the notebooks, there are some requirements for the libraries that needs to be fulfilled, therefore there is a txt file stating these requirements*

**Following outline of the notebooks:**
- Importing libraries 
- Hyperparameters
- Loading the data
- Creating the generator for 64x64 pictures
- Creating the discriminator
- Weights initialization
- **DCGAN model**
- Plotting the loss function. 
- Showing a real picture and the latest created fake picture. 
- Showing the evolution of the generated pictures. 

**Variations:** In the 3 notebooks, the code for the generator and discriminator classes differens in depth, according to the sizes that they produce.  
