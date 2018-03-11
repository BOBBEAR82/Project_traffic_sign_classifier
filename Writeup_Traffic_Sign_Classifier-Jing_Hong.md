# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./Writeup_images/Visualization.png "Visualization"
[image3]: ./Writeup_images/New_images.png "New_images"
[image4]: ./Writeup_images/New_images_prediction.png "New_images_prediction"
[image5]: ./Writeup_images/Visualizing_NN.png "Visualizing_NN"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/BOBBEAR82/Project_traffic_sign_classifier/blob/master/Traffic_Sign_Classifier-Jing_Hong.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32*32*3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data distributed in each sign.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. 

As a first step, I decided to convert the images to grayscale because after experiment and compare, I found that the training accuracy will be higher with grayscaled image.

As a last step, I normalized the image data because after experiment and compare, I found as well that the training accuracy will be higher after normalizing the images before input it to the network.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image   					| 
| Convolution 1     	| 1x1 stride, valid padding, outputs 28x28x10 	|
| RELU					|												|
| Convolution 2     	| 1x1 stride, valid padding, outputs 24x24x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 12x12x32 				|
| Convolution 3     	| 1x1 stride, valid padding, outputs 8x8x40 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 4x4x40 					|
| Flatten   	      	| Outputs 640 					|
| Fully connected 1		| Outputs 120  									|
| RELU					|												|
| Fully connected 2		| Outputs 84  									|
| RELU					|												|
| Fully connected 3		| Outputs 43  									|
 

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I am using an AdamOptimizer to optimize the loss of cross entropy. I am using a batch size of 128, epochs of 20, learning rate of 0.001. I tried to use different learning rate and epochs, and found the learning rate of 0.001 is still a good enough value in this case. And found epochs of 20 is enough to get expected accuracy, higher epochs may help to increase training accuracy a little bit, but may cause over fitting.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. 

My final model results were:

* validation set accuracy of 0.952 
* test set accuracy of 0.934

Compared with the original LeNet architecture, I added anoter convolution layer. My idea is another convolution layer can help to differentiate the features in different traffic sign, which can improve the prediction accuracy. From experiments, I found the training accuracy was improved from 0.89 to 0.96, and test accuracy was improved to >0.93.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are eight German traffic signs that I randomly picked from the German traffic signs test dataset:

![alt text][image3]

The most difficult one of these final picked images are the 3rd one, because it is very dark and hard to tell the details in the sign.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set 

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Bycycles crossing		| Bycycles crossing   									| 
| Keep right     			| Keep right 										|
| No passing for vehicles over 3.5 metric tons	| No passing for vehicles over 3.5 metric tons				|
| Child crossing	      		| Child crossing					 				|
| Speed limit (30km/h)			| Speed limit (30km/h)      							|
| Speed limit (100km/h)			| Speed limit (100km/h)      							|
| Right-of-way at the next intersection			| Right-of-way at the next intersection      							|
| Turn right ahead			| Turn right ahead      							|

The model was able to correctly guess 8 of the 8 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 93%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

Below is the result of top five soft max probabilities of each images picked.

![alt text][image4] 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

Below is the visualization of the net work with 2nd convolution layer.

![alt text][image5] 

### Suggest possible improvements
I think the main improvment I need is not in the code itself, but a better understanding how the neural network works, how it can learn by itself to classify differnt features in different complexities in different layer. That will help me to improve my knowledge in how to select propriate techniques when designing the network in different application. I plan to spend more time to walk through the TensorFlow library to achieve this goal. It will be appreciated if some useful guilds or links can be provided by reviewer.

