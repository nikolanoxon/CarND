# **Traffic Sign Recognition** 

## Writeup

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

[image1]: ./examples/GST_histogram.png "Histogram of Sign Classes in the Dataset"
[image2]: ./examples/preprocess.png "Preprocessing"
[image3]: ./examples/augment.png "Augmenting"
[image4]: ./examples/GST_01.jpg "Traffic Sign 1"
[image5]: ./examples/GST_01.jpg "Traffic Sign 2"
[image6]: ./examples/GST_01.jpg "Traffic Sign 3"
[image7]: ./examples/GST_01.jpg "Traffic Sign 4"
[image8]: ./examples/GST_01.jpg "Traffic Sign 5"
[image9]: .ecamples/original.png "Original"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup

My project code can be found here: [project code](https://github.com/nikolanoxon/CarND/tree/master/T1P2_traffic_sign_recognition/)

### Data Set Summary & Exploration

#### 1. Data Set Summary

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is [32px, 32px, 3ch]
* The number of unique classes/labels in the data set is 43

#### 2. Dataset visualization.

Shown below is a histogram of all the signs in the data set, binned according to their class.

![alt text][image1]

This provided a clear understanding of the variety of available examples for training.

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

#### Step 1: Preprocessing

Prprocessing of the data was performed with two methods:
1. Convert to greyscale
2. Normalize

Greyscaling was beneficial because it reduced the image channels from 3 to 1, effectivly reducing the number of inputs by 1/3. It was justified in that, for traffic signs, color does not add meaningful value this specific kind of object detection. Traffic signs are designed to stimulate "glance value" via shape and color for a human observer. However a CNN does not require the glance value benefit of color. And in fact all traffic signs are readily distinguishable from each other in B&W.

Normalization was performed second so that the mean and variance of each input would be 0 and 1 respectivly. This normalization of the inputes allows the weights to have similar values, which in turn improves the ability of the CNN to learn.

All preprocessing was done with OpenCV methods. Here is an example of a traffic sign image before and after preprocessing.

![alt text][image9]
![alt text][image2]

#### Step 2: Augmentation

Per the methods found in [LeCann's Paper on Traffic Sign Recognition]['http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf'], I also included augmented data to the original data set. Intuitivly this makes sense. By adding random perturbations to the original image, the robustness of the CNN to see identify signs in different situations is improved. The addition of augmented data resulted in significant improvement of the validation accuracy (see Section 4). The augmentations used were the following:
- X and Y translation of +/-2px
- Rotation about the center of +/-15deg
- Skew of the X and Y axes by +/-2px

![alt text][image9]

![alt text][image3]


#### 2. Network Architecture

The network used was derived from LeNet5, with improvements made to identify traffic signs. The changes from LeNet are:
- Changing the FC1 and FC2 outputs from (120, 84) to (250, 120)
- Adding a dropout layer after FC1 and FC2

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Gray image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, valid padding, outputs 14x14x6 	|
| Convolution 10x10    	| 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, valid padding, outputs 5x5x16 	|
| Fully connected		| outputs 250  									|
| Dropout       		| 50%       									|
| Fully connected		| outputs 120  									|
| Dropout       		| 50%       									|
| Fully connected		| outputs 43  									|
| Softmax				|           									|
|						|												|
|						|												|


#### 3. Network Training

Training was done using the Adams Optimizer, which was instructed to minimize the reduced mean of the softmax cross entropy between the logits and the labels. Adam was selected over traditional gradient decent because of its use of momentum to adjust the learn rate. This has been demonstrably shown to improve convergence rates. After much tuning (see section 4), the following hyperparameters were used:
- Learning Rate = 0.001
- Epochs = 50
- Batch Size = 50

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.


My final model results were:
* training set accuracy of 99.2%
* validation set accuracy of 97.1%
* test set accuracy of 95.1%

##### Initial Setup

|EPOCHS | BATCH_SIZE | rate | network | activation | normalization | augmentation | Validation | Test |
|------|------|------|------|------|------|------|------|------|
|10 | 128 | 0.001 | LeNet5 | ReLU | none | none | 0.863 | 0.853|

##### Preprocessing
|EPOCHS | BATCH_SIZE | rate | network | activation | normalization | augmentation | Validation | Test |
|------|------|------|------|------|------|------|------|------|
| 10 | 128 | 0.001 | LeNet5 | ReLU | simple | none | 0.885 | 0.891 |

##### Tuned Hyperparameters

##### Architecture Update

##### Augmentation
|EPOCHS | BATCH_SIZE | rate | network | activation | normalization | augmentation | Validation | Test |
|------|------|------|------|------|------|------|------|------|
| 100 | 50 | 0.001 | LeNet5+ | ReLU | OpenCV-MinMax | rotate/translate/skew | 0. | 0. |

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][Traffic Sign 1] ![alt text][Traffic Sign 2] ![alt text][Traffic Sign 3] 
![alt text][Traffic Sign 4] ![alt text][Traffic Sign 5]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


