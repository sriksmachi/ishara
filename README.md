# ishara
Gesture Detection using Deep Learning

## Problem Statement 

Detect gestures from sequence of images generated from a video. There are five gestures - swipe left/right, thumbs up/down and stop. 
Each video/frame is taken from different device, hence the resolution is different. 


## Solution Options

Since the data is a sequence of frames we cannot use a conventional CNN, there are 2 options to deal with such data. 
1. CNN + RNN - Extract the features from each frame, create a sequence of features and feed to RNN to learn from the sequence and predict. 
2. 3D CNN - Convolve using 3 layered kernel, which is just an extension of 2D convolution. 

We will try both the options

## Solution 1 - 3D Convolution

### Preprocessing 

The following steps will be taken for the experiment
 
- Load the data and visualize, get a feeler of how the images look like. 
- Understand what data transformations / augmentation techniques can be applied to make the model more robust. 
- Create a data generator
- Use the data generator to create training and validation sets. 
- Create a full fledged model
- Run ablation experiment to see if the model is working
- Try to overfit the model using more epochs to validate if the loss is decreasing and the accuracy is incresing. 
- Define callbacks for saving the model based on some criteria, adjust learning rate etc. 
- Tune the hyper parameters and re-run the model until desired accuracy is aquired. 




