
**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/Example.png
[image2]: ./output_images/bin_spatial.png
[image3]: ./output_images/histogram.png
[image4]: ./output_images/hsv.png
[image5]: ./output_images/hog.png
[image6]: ./output_images/pipeline.png
[image7]: ./output_images/slide.png
[image8]: ./output_images/search.png
[image9]: ./output_images/heatmap.png
[video1]: ./output/project_video_output.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### README

### Feature Extraction

#### 1. Load and Visualize Data

The code for this step is contained in the 2nd, 3rd and 4th code cells of the IPython notebook.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

#### 2. Pipeline Definition

I choose to define a pipeline to extract features from every image of both datasets.  
The pipeliene was defined like that:
* Color Space
* Bin Spatial
* Color Histogram
* HOG

##### 2.1 Bin Spatial

The code for this step is contained in the 5th code cell of the IPython notebook.

First, I implemented the Bin Spatial. Here all the image are resized to 32x32 pixels and then tranformed into a 1-D vector.
```
# Resize images to 32x32 pixels
def bin_spatial(img, size=(32, 32), vis=False):
    
    if not vis:
        # Use cv2.resize().ravel() to create the feature vector
        features = cv2.resize(img, size).ravel() 
        # Return the feature vector
        return features
    else:
        img = cv2.resize(img, size)
        features = img.ravel() 
        #Return the feature vector and the image for visualization
        return features, img
```

The result can be seen on the next image:

![alt text][image2]

##### 2.2 Color Histogram

The code for this step is contained in the 7th code cell of the IPython notebook.

Second, I implemented the color histogram feature extraction. Here each channel of the image is converted into a histogram and all three of them are further converted to a single 1-D vector.
```
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the RGB channels separately
    rhist = np.histogram(img[:,:,0], nbins, bins_range)
    ghist = np.histogram(img[:,:,1], nbins, bins_range)
    bhist = np.histogram(img[:,:,2], nbins, bins_range)
    # Generating bin centers
    bin_edges = rhist[1]
    bin_centers = (bin_edges[1:] + bin_edges[0:len(bin_edges)-1])/2
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((rhist[0], ghist[0], bhist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return rhist, ghist, bhist, bin_centers, hist_features
```
The result can be seen on the next image:

![alt text][image3]

##### 2.3 Exploring Color Spaces

The code for this step is contained in the 9th code cell of the IPython notebook.

Third, I chose to test different color spaces and then decide which one of them highlighted the cars the best. After iterating through a list of color spaces, I found that HSV looked like the best option.
```
def cvt_color(image, cspace='RGB'):
    if cspace != 'RGB':
        if cspace == 'HSV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif cspace == 'LUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
        elif cspace == 'HLS':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        elif cspace == 'YUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    else: feature_image = np.copy(image) 
    return feature_image
```
An example of how the cars and non-cars look like in HSV color space can be seen next:

![alt text][image4]

##### 2.4 HOG

The code for this step is contained in the 11th code cell of the IPython notebook.

Finally, the HOG extraction. For this part, I based myself on an online article (https://medium.com/@mohankarthik/feature-extraction-for-vehicle-detection-using-hog-d99354a84d10) to decide which where the best parameters to use on my HOG extraction.

```
def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), block_norm= 'L2-Hys',
                                  transform_sqrt=True, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), block_norm= 'L2-Hys',
                       transform_sqrt=True, 
                       visualise=vis, feature_vector=feature_vec)
        return features
```

So, I chose to use `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`, here is an example of how it looks like:

![alt text][image5]


##### 2.5 Final Pipeline

The code for this step is contained in the 13th code cell of the IPython notebook.

So, my final pipeline does the following:
* Read an image
* Convert color
* Bin spatial
* Color histogram
* HOG

```
def extract_features(imgs, cspace='RGB', spatial_size=(32, 32),
                        hist_bins=32, hist_range=(0, 256), orient=8, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for img in imgs:
        # Read in each one by one
        img = mpimg.imread(img)
        # apply color conversion if other than 'RGB'
        feature_image = cvt_color(img, cspace)
        # Apply bin_spatial() to get spatial color features
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        # Apply color_hist() to get color histogram features
        _,_,_,_,color_features = color_hist(feature_image, nbins=hist_bins, bins_range=hist_range)
        # Apply get_hog_features() to get HOG features
        # Call get_hog_features() with vis=False, feature_vec=True
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features)        
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        # Append the new feature vector to the features list
        features.append(np.concatenate((spatial_features, color_features, hog_features)))
    # Return list of feature vectors
    return features
```

Next, you can see the plots of the features from a Vehicle and from a Non-Vehicle image:

![alt text][image6]

Analysing the plots visually, we can already notice that there are differences between the features of the images, so that can be a good indicator that the classifier will perform well.

#### 3. Classifier

After extracting all the features from all the images, as we can see on cells number 15 and 16 from the Jupyter Notebook. We defined all of our features X and all of our labels y.

With that, we could split our data between train and test sets:
```
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rand_state)
```

We trained a normalizer on our X_train dataset and apply it both in our X_train and X_test.
```
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X_train)
# Apply the scaler to X
X_train = X_scaler.transform(X_train)
X_test = X_scaler.transform(X_test)
```
The normalization proved to be efficient as we calculated the mean and standard deviation of the features:
```
X_train mean: -6.765167911465336e-18
X_train standard deviation: 0.9959266633137144
```

For our classifier we chose to work with the `clf = SVC()`, that has a `rbf` kernel. And achieved the following results:
```
388.21 Seconds to train SVC...
Test Accuracy of SVC =  0.9913
My SVC predicts:  [0. 1. 1. 0. 1. 1. 0. 1. 1. 1.]
For these 10 labels:  [0. 1. 1. 0. 1. 1. 0. 1. 1. 1.]
0.1998 Seconds to predict 10 labels with SVC
```

### Sliding Window Search

To perform the sliding window search, first, we implemented the sliding window function, in `slide_window()`, where we defined the whole area that should be searched.  
The area can be seen next:

![alt text][image7]

With the area defined, we started to search for vehicles inside that area with `search_windows()`. The result:

![alt text][image8]

To reduce the multiple boxes to a single one and remove the false positives, we used a combination of `heat_map` and `labels`, the results achieved can be seen next:

![alt text][image9]

All the code for this part can be easily find on the highlighted section `Test Classifier on Images` inside the Notebook.

### Video Implementation

Here's a [link to my video result](./project_video.mp4)

On my Video Implementation, I tried to reduce the wobbly effect by saving the previous detections on a class called `Vehicle_Detected`. 

```
# Define a class to store data from video
class Vehicle_Detect():
    def __init__(self):
        # history of boxes from previous frames
        self.prev_boxes = [] 
        threshold = 15
        
    def add_boxes(self, box):
        self.prev_boxes.append(box)
        if len(self.prev_boxes) > threshold:
            # throw out oldest rectangle set(s)
            self.prev_boxes = self.prev_boxes[len(self.prev_boxes) - threshold:]
```

With all the previous boxes that contained cars being saved, the result can be used as a threshold and also to make the boxes transitions smoother.

```
for boxes in det.prev_boxes:
        heat = add_heat(heat, boxes)

    heat = apply_threshold(heat,  1 + len(det.prev_boxes)//2)
```

This part of the code can be found on the last 3 code cells of the Notebook.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Considering the approach that I took, of using the previously described pipeline, one of the major problems that I encountered was with the left side of the screen. I tried augmenting my dataset, tried using different classifiers, tried different search techniques, but I always kept finding false positives, specially on the shadowed areas. To solve that I defined a shorter area of interest. But for future works, I believe that working with Deep Learning, specially YOLO algorithms would be the best way to improve the results and make it more robust. 

