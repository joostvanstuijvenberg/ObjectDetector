# ObjectDetector
An enhanced version of OpenCV's SimpleBlobDetector

## Setting up detection parameters

### Programmatically
It is very well possible to create an ObjectDetector instance and configure it in your code. 

```cpp
// Start by creating an ObjectDetector instance. Give it a minimum distance between BLOBs of 10.
ObjectDetector od(10.0);

// Create an object that encapsulates the desired thresholding algorithm. Use the range algorithm.
auto tra = std::make_shared<ThresholdRangeAlgorithm>(40, 150, 10, 3);
od.setThresholdAlgorithm(tra);

// Create an object that encapsulates the desired filter. Filter out objects with an area within specified limits.
od.addFilter(std::make_shared<AreaFilter>(4000, 50000));

// Now let the object detector do its job and receive the keypoints of all detected objects.
auto keypoints = od.detect(image);

```

### In an xml file


## Benefits over SimpleBlobDetector
- features multiple threshold algorithms, including Otsu's
- user defined filter classes
- findContours uses CHAIN_APPROX_SIMPLE (as opposed to SimpleBlobDetector, which uses CHAIN_APPROX_NONE)
- filtering by color actually uses a range of colors