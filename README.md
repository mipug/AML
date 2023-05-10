# Project in the course: Advanced Machine Learning for Data Science

## I’m Something of a Painter Myself
*To what extent is it possible to identify and create pictures that resemble Monet painting-style using DCGan (Deep Convolutional generative adversarial network)?*

This project is inspired by a Kaggle competition - https://www.kaggle.com/competitions/gan-getting-started/overview 
The problem should be solved with a variant of a GAN model. We have decided to implement a DCGan to solve the problem and answer our own created research question. Creating a DCGan model we are working in the unsupervised machine learning paradigm. 

The dataset provided in the Kaggle competition consists of ~300 Monet paintings and ~7000 photos, both in jpg and TFRecord format. 

**Goals:**
- Generate 7000 pictures in the style of Monet
- In the early stages of the DCGAN process, we have a Discriminator goal as defined: the discriminator is able to identify a fake Monet and a real Monet. 
- In the later stages of the DCGAN process, we have a generator goal defined as Generating a picture that passes the discriminator as a real Monet painting. 


**Github repository:**

This github repository contains 3 different notebooks, each containing 3 different DCGAN models. The models varies in the sizes of the pictures that they work with, they are producing the repective picture sizes: 64x64, 128x128, 256x256. 

In each of the notebooks there is various hyperparameters that can be modified, to change the outcome of the models. 

*The solution and notebook is inspired by Pytorch toturial on DCGAN:* https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html 

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
