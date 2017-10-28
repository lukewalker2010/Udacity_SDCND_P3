#**Behavioral Cloning**

##Luke Walker
##10/28/17

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/model.png "Model Visualization"
[image2]: ./examples/center.png "Center"
[image3]: ./examples/left.png "left"
[image4]: ./examples/right.png "Right"
[image5]: ./examples/loss.png "Loss"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md summarizing the results, you're reading it

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model_nNvidia_modified.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

additionally model_experiment ipython notebook was used to generate images and plots for this writeup.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network. First I have a lambda layer to normalize the data. Then I crop off the top 50 pixels and bottom 20 pixels. this is then followed up by 5 convolution layers. These layers includes RELU to introduce non-linearity. After those layers I flatten the model and then have 4 dense layers. This can all be seen in the model summary below.

####2. Attempts to reduce overfitting in the model

I only trained my model for 3 epochs to try to eliminate overfitting. I noticed after 3 epochs the loss started to increase.

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually. I used a total of 24108 images. Of those 19286 were used in training and 4822 were used for validation.

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I mainly used combination of center lane driving, but also added some recovering from the left and right sides of the road as well as driving the track in reverse.

I trying converting the images to greyscale but found that the car did not drive as well after so I stuck with color images.

For details about how I created the training data, see the next section.

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the LeNet architecture I thought this model might be appropriate because it can be trained quickly and verify that my code is working.

From there I decided to use the nVidia model that was found here: https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/

I added in a cropping layer to get rid of the sky and hood of the car as those sections of the image don't contribute to the model.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat the overfitting, I modified the model so that I was only running 3 epochs.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. to improve the driving behavior in these cases, I added more training data.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 108-131) consisted of a convolution neural network.

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first started with the images already recorded. I then added recorded two laps on track one using center lane driving and one reverse lap. Here is an example image of center lane driving with the center left and right cameras:

![alt text][image2]
![alt text][image3]
![alt text][image4]

To augment the data sat, I also flipped images and angles thinking that this would help the model since most of the turns are left the car would not know how to turn right. I used cv2.filp in the generator to flip the images.

After the collection process, I had 24108 images. I randomly shuffled the data set and put 20% of the data into a validation set. This equated to 19286 training images and 4822 were used for validation.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by an increase in loss with any additional epochs. I used an adam optimizer so that manually training the learning rate wasn't necessary.

A plot of the loss can be seen below:

![alt text][image5]

The video of driving the track was created using video.py is the run1.mp4 file.

I enjoyed this project. I think that I could make my model more robust by trying various preprocessing techniques and possibly switching color spaces to eliminate the effect of exposure on the model.
