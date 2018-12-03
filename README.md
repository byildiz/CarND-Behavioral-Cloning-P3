# **Behavioral Cloning Project**

The goals / steps of this project are the following:

* Use the simulator to collect data of good driving behavior
* Build a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./images/loss_without_dropout.png "Loss graph without dropout layers"
[image2]: ./images/loss_with_dropout.png "Loss graph with dropout layers"
[image3]: ./images/model_with_dropout.png "Model with dropout layers"
[image4]: ./images/straight1.gif "Normal driving"
[image5]: ./images/turn1.gif "Normal driving 2"
[image6]: ./images/recovery1.gif "Recovery 1"
[image7]: ./images/recovery2.gif "Recovery 2"

## Rubric Points

Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.

---

### Required Files

#### 1. The submission includes a model.py file, drive.py, model.h5 a writeup report and video.mp4.

My project includes the following files:

* [model.py](./model.py) containing the script to create and train the model
* [drive.py](./drive.py) for driving the car in autonomous mode
* [model.h5](./model.h5) containing a trained convolution neural network
* [writeup.md](./writeup.md) summarizing the results (this file)

### Quality of Code

### 1. The model provided can be used to successfully operate the simulation.

The trained model (model.h5) can safely drives the car in the track one. The model can be tested with above command:

```sh
python drive.py model.h5
```

#### 2. The code in model.py uses a Python generator, if needed, to generate data for training rather than storing the training data in memory. The model.py code is clearly organized and comments are included where needed.

All required functions from reading data to training model are given with their explanations in [model.py](./mode.py).

At first, I choose to read all data at once. It worked because I only used the images comes from the center camera. But, then, I started to use left and right images with their flipped versions and I got out-of-memory error. Thus, I implemented `batch_generator` function to simultaneously load data batch by batch while training.

### Model Architecture and Training Strategy

#### 1. The neural network uses convolution layers with appropriate filter sizes. Layers exist to introduce nonlinearity into the model. The data is normalized in the model.

At first, I used a LeNet-based CNN architecture and trained a model. However, it could not keep the car on the track at some points and I decided to use a more powerful architecture more specifically like the architecture described in [this paper](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)

The implemented architecture has a lambda layer which normalizes the input and 5 convolution layers, the first 3 of them uses 5x5 convolution kernels with 2x2 strides and the rest uses 3x3 kernels without stride. After each convolution layer, **ReLU** activation function is applied for nonlinearity.

The data normalization is embedded in the model since the simulation gives RGB camera images without normalization and those images also need same normalization process. Another solution can be to implement same normalization function in [drive.py](./drive.py) but it may need to be updated each time the normalization parameters are changed.

#### 2. Train/validation/test splits have been used, and the model uses dropout layers or other methods to reduce overfitting.

3 dropout layers with a drop rate **0.25** were inserted into implemented CNN architecture to reduce the overfitting.

Two loss to epoch graph are given in the following figure. The data of the first graph was collected before inserting dropout layers and the second data was collected after inserting dropout layers ([model.py](./model.py) lines 165, 168, and 172).

|Before Dropout|After Dropout|
|--------------|-------------|
|![image1]     |![image2]    |

The model was trained and validated on different data sets to ensure that the model was not overfitting. Since generators were used, first, whole data set was divided into two data sets with a size ratio 4:1, train and validation sets respectively, and then two generators were created ([model.py](./model.py) line 199). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Learning rate parameters are chosen with explanation, or an Adam optimizer is used.

Adam optimizer was used so that the learning rate was not tuned manually ([model.py](./model.py) line 207).

#### 4. Training data has been chosen to induce the desired behavior in the simulation (i.e. keeping the car on the track).

Training data was chosen to keep the vehicle driving on the road. Besides the given data set, I also recorded additional data, one lap normal direction, one lap reverse direction and several rescuing actions from both left and right sides.

### Architecture and Training Documentation

#### 1. The README thoroughly discusses the approach taken for deriving and designing a model architecture fit for solving the given problem.

The problem is to calculate the appropriate steering angle for a given image taken from the simulator to keep the car on track without leaving the track. My approach consists of 3 main steps which are basic implementation, improving model and testing.

The first step is to implement a training pipeline with a most basic model architecture without considering model quality. More specifically, provided data was split into a training and a validation set and a one-layer neural network was trained using only center camera images. Since, the output is a value representing steering angle, mean squared error function was used as loss function and the model had high loss on both train and validation sets.

The second step is improving the model quality in terms of loss. First, I used a well known model architecture to improve model. I chose LeNet and modified sizes of first layer and output to fit our problem. LeNet decreased the loss on both train and validation sets but the result was not quite good.

To improve the result, then, I decided to use flipped version of center images, left and right camera images with their flipped versions. However, I encountered lack of memory problem. To overcame the problem, I used generators. I also added a cropping layer to the model to crop out unrelated portions of camera images in addition to a normalization layer. These additions also improved the loss but it was also not good as expected.

I, then, implemented a more advanced model architecture as described model in Nvidia paper. This is also decreased the losses but this time there is a gap between train and validation loss. This is the indicator of overfitting.

To combat the overfitting, I modified the model so that I added 3 dropout layers and this was solved the problem and validation loss decreased as train loss.

The graphs of loss to epoch before and after dropout are given in the previous section and they shows the improvement.

I expected this last model would keep the car on track but it did not. The problem was that car was wobbling side to side and at some points the car rolled down from the road. I thought that it was because of correction angle (it was **0.2**) used for left and right camera images. It could be too high.

I tried to decrease the correction angle and it improves driving and found an optimal value which is **0.07** for my setup. Further more, I recorded some additional data from track one and trained the final model. The final model can keep the car on track 1 without leaving drivable postion of the road with less wobbling.

#### 2. The README provides sufficient details of the characteristics and qualities of the architecture, such as the type of model used, the number of layers, the size of each layer. Visualizations emphasizing particular qualities of the architecture are encouraged. Here is one such tool for visualization.

The final model architecture consisted of a copping layer, a normalization layer, 5 convolutional layers followed by relu activation function for nonlinearity, 4 fully connected layers, and 3 dropout layers. The dropuput layers inserted at after the first 3 convolutional layers, end of all convolutional layers, and before the last fully connected layer. Input images has a shape of **160x320x3** and cropping layer crops top **70** pixels and bottom **25** pixels out because they contains much irrelevant data. The normalization layer is the implementation of Lambda layer of Keras which scales each channel value between **-0.5** and **0.5**. Each of the dropout layers has a dropout rate of **0.25**.

The first 3 convolutional layers uses **5x5** kernels with **2x2** strides and have depths of **24**, **36**, and **48**, respectively. Remaining convolutional layers use **3x3** kernels with and have depths of **64** each. All of them use **valid** padding.

After convolutional layers, extracted features are flattened. The 4 connected layers have **100**, **50** and **10** outputs and **1** output, respectively. This final single output would be predicted steering angle.

Here is a visualization of the final architecture created by Keras's own model visualization API.

![image3]

#### 3. The README describes how the model was trained and what the characteristics of the dataset are. Information such as how the dataset was generated and examples of images from the dataset must be included.

Although the provided data was adequate for training of a successful model, I also recorded some extra data. My record consisted of two laps and several recoveries from left and right sides of the road back to center. The below gif images are from my recording:

![image4]
![image5]
![image6]
![image7]

The first two illustrates good driving behavior (Absolutely I am not a good video game player) and following two are for teaching model how to recover from bad situations.

To increase data points, I used augmentation. For each data point, 6 variations (center and flipped center images, left and flipped left images and right and flipped right images) were created. Thus, from **1674** data points, I got **64044** images.

The steering angles of flipped center images were calculated by negating original steering angle. The steering angles of left and right images were calculated by subtracting and adding correction angle from and to original steering angles, respectively. The steering angles for flipped versions of left and right images were calculated first correcting steering angles as described and then negating the corrected angles. The correction angle was obtained empirically. For my setup, it was **0.07**.

Validation ratio was selected as **0.2** so sizes of train and validation sets would be **51236** and **12808**, respectively. The validation set helped to determine if the model was over or under fitting.

I added a cropping layer to crop out less irrelevant portions of the images to cut out the noisy parts. It helped to train better model. Top **70** and bottom **25** pixels were determined as irrelevant empirically. Cropping also helped to reduce memory usage.

Preprocessing is embedded into model not to add normalization function to not to [drive.py](./drive.py) and not to update [drive.py](./drive.py) each time normalization function changed.

The number of epochs was **10**. It may or may not be ideal number but it was enough to overfitting. At this point, I used a model checkpoint saver to save best model according to validation loss to select best model at the end of training.

I also used an adam optimizer not to tune learning rate manually.

### Simulation

#### 1. No tire may leave the drivable portion of the track surface. The car may not pop up onto ledges or roll over any surfaces that would otherwise be considered unsafe (if humans were in the vehicle).

A [video.mp4](./video.mp4) file was provided to show how well the car was driven on track 1 by the final model. As seen in the video, the car was driven with a speed of 30 mph without leaving drivable area for two laps.