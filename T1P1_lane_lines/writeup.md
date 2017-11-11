# **Finding Lane Lines on the Road** 

## Writeup

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./solidWhiteCurve.jpg "Example from Video 1"
[image2]: ./solidYellowLeft.jpg "Example from Video 2"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

The pipeline I developed was based on the lesson material, which I then optimized to perform on the provided images and videos. I used two functions (in addition to the helper functions) to accomplish this: pipeline() and draw_lines()


##### The following steps were performed in the pipeline() function for each frame

1. Convert the frame to grayscale
2. Apply a Gaussian blur (k=5) to smooth the image
3. Apply a Canny transform to find all edges
3. Mask the frame
4. Apply a Hough transform to obtain lane line candidates

##### The following steps were performed in the draw_lines() function for each frame
5. Filter out the candidates with implausible slopes (too shallow)
6. Calculate the slope and intercept of each candidate. This assumed all lines follow the Y = mX + b forumulation.
7. Utilize K-means clustering with a dataset of each candidate's x_min, y_min, x_max, y_max, slope, and intercept to identify the index of each candidate (left or right) and the centroid of each lane.
8. Use the centroid information to find the dimensions of each lane line. The bottom of the lane was assumed to have a Y value equal to the Y dimension of the image (that is, it extended to the bottom of the frame). The Y value of the top of the line was taken as the max Y value of all line candidates belonging to the left lane (as identified by k-means). The max and min X values were extrapolated using these Y values with the centroid values for slope and intercept.
9. Draw the left and right lanes over the original image using the values found in the previous step.

#### Examples

![alt text][image1]

![alt text][image2]


### 2. Identify potential shortcomings with your current pipeline

Several shortcomings could impact the performance of this algorithm, and in fact the current implementation crashes when applied to the challange video. For more complicated road types, this algorithm would fail for some of the following reasons:

1. Heavy shadows tricking the Hough transform into incorrectly identifying lane lines.
2. Vertical lines have infinite slope and no Y-intercept. Infinity and NaN cannot be used for k-means because a centroid cannot be found.
3. Linear lines do not fit well to lanes with small curvature.
4. K-means requires a number of centroids to be predefined. For more complicated lane line algorithms, an indeterminate number of lanes may be desired to be known.
5. Lines are jittery and do not take into account their history, this creates unrealistic jumps.
6. Inclimate weather and unmaintained roads will return false lanes (road ruts, tar strips) or no lanes.
7. Onramps/offramps are not accounted for.
8. Lane color/type is not accounted for. This might be needed for passing scenarios. 
9. Instead of tracking the innermost part of the lane line, the average is taken. This would affect the ability to stay within the lanes.

### 3. Suggest possible improvements to your pipeline

For each shortcoming, here are possible solutions:

1. Using a more robust edge finder and color filtering to ifnore shadows.
2. Transform the lines into a exression that does not include infinities (polar)
3. Use a polynomial interpolation instead of linear.
4. Forgo K-means in favor of a leaner and more flexible sorting algoirthm (regression?)
5. Use a moving average to smooth the lanes.
6. Same as 1.
7. Develop a means of detecting onramps/offramps (possibly by noticing large differences in lane curvature, possibly by machine learning)
8. Run Canny/Houghs transforms on yellow and white masked images seperately. Use Houghs to see if a lane is made of many or few segments (striped or solid lines).
9. Use the outer edges for confirmation but plot the line along the inner edges.