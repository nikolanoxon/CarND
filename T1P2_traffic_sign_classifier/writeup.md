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
[image4]: ./examples/GST_01.jpg "Speed Limit 30kph"
[image5]: ./examples/GST_09.jpg "No Passing"
[image6]: ./examples/GST_12.jpg "Priority Road"
[image7]: ./examples/GST_15.jpg "No Vehicles"
[image8]: ./examples/GST_25.jpg "Road Work"
[image9]: ./examples/original.png "Original"
[image10]: ./examples/visualize_cnn.png "FC1 Visualization"
[image11]: ./examples/softmax_01.png "Softmax Speed Limit 30kph"
[image12]: ./examples/softmax_09.png "Softmax No Passing"
[image13]: ./examples/softmax_12.png "Softmax Priority Road"
[image14]: ./examples/softmax_15.png "Softmax No Vehicles"
[image15]: ./examples/softmax_25.png "Softmax Road Work"

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

#### 1. Data Preperation

##### Step 1: Preprocessing

Prprocessing of the data was performed with two methods:
1. Convert to greyscale
2. Normalize

Greyscaling was beneficial because it reduced the image channels from 3 to 1, effectivly reducing the number of inputs by 1/3. It was justified in that, for traffic signs, color does not add meaningful value this specific kind of object detection. Traffic signs are designed to stimulate "glance value" via shape and color for a human observer. However a CNN does not require the glance value benefit of color. And in fact all traffic signs are readily distinguishable from each other in B&W.

Normalization was performed second so that the mean and variance of each input would be 0 and 1 respectivly. This normalization of the inputes allows the weights to have similar values, which in turn improves the ability of the CNN to learn. All preprocessing was done with OpenCV methods.

##### Step 2: Augmentation

Per the methods found in [LeCann's Paper on Traffic Sign Recognition]('http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf'), I also included augmented data to the original data set. Intuitivly this makes sense. By adding random perturbations to the original image, the robustness of the CNN to see identify signs in different situations is improved. The addition of augmented data resulted in significant improvement of the validation accuracy (see Section 4). All augmentations used were performed using OpenCV using the following:
- X and Y translation of +/-2px
- Rotation about the center of +/-15deg
- Skew of the X and Y axes by +/-2px

Original
![alt text][image9]

Preprocessed
![alt text][image2]

Augmented
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

#### 3. Network Training

Training was done using the Adams Optimizer, which was instructed to minimize the reduced mean of the softmax cross entropy between the logits and the labels. Adam was selected over traditional gradient decent because of its use of momentum to adjust the learn rate. This has been demonstrably shown to improve convergence rates. After much tuning (see section 4), the following hyperparameters were used:
- Learn Rate = 0.001
- Epochs = 50
- Batch Size = 50

#### 4. Solution Approach

My final model results were:
* training set accuracy of 99.2%
* validation set accuracy of 97.1%
* test set accuracy of 95.1%

LeNet5 was used as the baseline, per the recommendation of the course instructor. LeNet5 seemed applicable because it is a CNN used for symbolic evaluation, and LeNet5 had been modified and used for traffic sign recognition previously. The first step was to preprocess the data, resulting in the following:

##### Initial Setup
|EPOCHS | BATCH_SIZE | rate | network | activation | normalization | augmentation | Validation | Test |
|------|------|------|------|------|------|------|------|------|
|10 | 128 | 0.001 | LeNet5 | ReLU | none | none | 0.863 | 0.853|

##### After Preprocessing
|EPOCHS | BATCH_SIZE | rate | network | activation | normalization | augmentation | Validation | Test |
|------|------|------|------|------|------|------|------|------|
| 10 | 128 | 0.001 | LeNet5 | ReLU | simple | none | 0.885 | 0.891 |

I felt the next step was to identify good hypterparameters. The hyperparameters were tuned by picking a default setting, and then adjusting each hyperparameter individually. Each tuning which resulted in the best validation accuracy without sacrificing unneccesary training speed was kept. The default setting was:
- Learn Rate = 0.001
- Epochs = 10
- Batch Size = 500

The range of tunings were:
- Learn Rate = 0.0001, 0.001, 0.005, 0.1
- Epochs = 5, 10, 20, 50, 100
- Batch Size = 10, 50, 128, 250, 500, 1000, 2000

##### Tuned Hyperparameters
|EPOCHS | BATCH_SIZE | rate | network | activation | normalization | augmentation | Validation | Test |
|------|------|------|------|------|------|------|------|------|
| 50 | 50 | 0.001 | LeNet5 | ReLU | simple | none | 0.955 | 0.930 |

After exhaustivly tuning the hyperparameters, the project requirement for validation accuracy had been fulfilled. There was many more options yet to explore, however.

First, dropouts were added after FC1 and FC2. These were not included in the original LeNet5, but since its inception have been shown to remarkedly improve robustness for CNNs. Dropouts are particularly useful for traffic sign recognition, because many times some of the features of a sign may be lost due to weather conditions, lighting, angle of approach, deterioration, etc.

Since LeNet5 was designed for numbers, it has relatively small fully connected layers. I thought that since traffic signs were so much more complex than numbers, there was likely many more features which could be extrapolated from the dataset. With that in mind, the size of FC1 and FC2 was doubled from 120 and 84 to 250 and 120. This allows the network to train on more features it observes. This has an added benefit when combined with dropouts, because the network is training with roughly the same number of features as before (2x the features, 50% dropout), but is also building robustness by training through different paths every pass through. These additions improved the test accuracy by over 1.5%.

##### Architecture Update + OpenCV Normalization
|EPOCHS | BATCH_SIZE | rate | network | activation | normalization | augmentation | Validation | Test |
|------|------|------|------|------|------|------|------|------|
| 50 | 50 | 0.001 | LeNet5 + 2xFC | ReLU + 50% drop | OpenCV | none | 0.961 | 0.947 |

After these changes it was observed that the training accuracy was 100%. This seemed to point towards overfitting of the training data, which was further corroborated when tested against externally chosen traffic signs. The last step was to add augmented data, as outlined in 1.1. This further built robustness by adding skewing, rotation, and translation of the images as well as doubling the size of the dataset.

##### OpenCV Augmentation
|EPOCHS | BATCH_SIZE | rate | network | activation | normalization | augmentation | Validation | Test |
|------|------|------|------|------|------|------|------|------|
| 100 | 50 | 0.001 | LeNet5+ | ReLU | OpenCV-MinMax | rotate / trans / skew | 0.971 | 0.951 |

### Test a Model on New Images

#### 1. External German Traffic Signs

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

Difficulties:
1. 30 kph sign could be mistaken with other speed signs due to the low resolution of the text
2. No Passing sign shares features with other signs with vehicles in them.
3. Priority Road: In poor lighting, the yellow could be indistinguishable from the white background
4. No Vehicles: Shares basic feature of white on red with many signs.
5. Road Work: Many fine details may be hard for the network to differentiate. Looks very similar to the children crossing sign.

#### 2. Discussion on external signs

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed Limit 30kph    	| Speed Limit 30kph   							| 
| No Passing     		| No Passing 									|
| Priority Road			| Priority Road									|
| No Vehicles   		| No Vehicles					 				|
| Road Work	    		| Road Work         							|

On the external traffic sign dataset, an accuracy of 80% was achieved. On other training rounds for this network, the accuracy was 100%. Compared to the test set (95.1%), this result matches up excellently. The only mistaken asignment was the 30 kph sign for a 40 kph sign. This intuitivly makes sense. They're both speed signs, and with the low resolution of the data it can be difficult to distinguish between the two.

#### 3. Softmax on external signs

The code for making predictions on my final model is located in the 28th cell of the Ipython notebook.

##### 30 kph sign #1
![alt text][image4] ![alt text][image11]

The network is 56% confident this is a 50 kph stop sign, and 24% confident it is a 30 kph sign. 50 kph is an understandable guess given that they are both speed signs and the similarity between 5 and 3.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .56         			| Speed Limit 50 kph							| 
| .24     				| Speed Limit 30 kph							|
| .19					| Speed Limit 80 kph							|
| 5.5e-5      			| Speed Limit 70 kph							|
| 9.7e-6			    | Speed Limit 60 kph							|

##### Road Work #25
![alt text][image8] ![alt text][image15]

The network correctly identified with 93% certainty that this is a road work sign. The other possibilities are also triangular signs with pictograms on them.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .93         			| Road Work   									| 
| .24     				| Wild Animals Crossing 						|
| .02					| Bicycles Crossing								|
| .01       			| Slippery Road			 						|
| .003  			    | Bumpy Road      								|

##### No Vehicles #15
![alt text][image7] ![alt text][image14]

The network was exceptionally confident that this was a No Vehicles sign. It is correct.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0					| No Vehicles									| 
| 2.3e-13  				| Speed Limit 70 kph							|
| 1.6e-13				| Priority Road									|
| 1.4e-15				| Stop					 						|
| 5.6e-16				| Speed Limit 50 kph							|

##### Priority Road #12
![alt text][image6] ![alt text][image13]

The network was exceptionally confident that this was a Priority Road sign. It is correct.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0					| Priority Road									| 
| 2.7e-27  				| Keep Right									|
| 9.9e-34				| Roundabout Mandatory							|
| 4.8e-35				| Speed Limit 100 kph	 						|
| 3.2e-37				| No Entry										|

##### No Passing #9
The network was exceptionally confident that this was a No Passing sign. It is correct.

![alt text][image5] ![alt text][image12]

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0					| No Passing									| 
| 1.5e-13  				| Speed Limit 120 kph							|
| 1.0e-13				| No passing for vehicles over 3.5 metric tons	|
| 1.9e-15				| No Vehicles			 						|
| 8.8e-16				| Speed Limit 70 kph							|

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

![alt text][image10]

For this sign, the network used the perimeter ring, the gradient between the interior and exterior colors, and the circular nature of the sign. In particular FeatureMap 3 focused strongly on the lack of darkness in the center circle, While FeatureMap1 and 5 focused on the solid white middle circle. FeatureMap2 seems to focus on the gradient between rings.