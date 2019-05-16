# ObjectDetector
ObjectDetector is an enhanced version of OpenCV's SimpleBlobDetector. See below for a list of benefits over this class.

## Setting up detection parameters
The object detector does its job according to its detection parameters. There are two ways to specify these parameters; programmatically and in an xml file. While the former method might be handy for a quick setup, the latter method is preferable since you can adjust the detection parameters without repetitively going through compilation and linking.

### Programmatically
It is very well possible to create an ObjectDetector instance and configure it in your code. Start by instantiating an object from the ObjectDetector class. It needs a minimum distance between objects. 10.0 is a good default.

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
You can load the complete definition (ie. threshold algorithm, minimum repeatability and the various filters) from an xml file. This file needs to have the following format:
```xml
<?xml version="1.0"?>
<opencv_storage>

  <thresholdAlgorithm>
    <ThresholdFixedAlgorithm>
      <threshold>150</threshold>
    </ThresholdFixedAlgorithm>
  </thresholdAlgorithm>

  <minDistBetweenObjects>10.</minDistBetweenObjects>

  <filters>
    <AreaFilter>
      <min>1000.</min>
      <max>10000.</max></AreaFilter>
  </filters>

</opencv_storage>
```

## Benefits over SimpleBlobDetector
- features multiple threshold algorithms, including Otsu's
- no need to use OpenCV's Ptr<SimpleBlobDetector> construct
- makes extensive use of smart pointers and move semantics
- user defined filter classes
- findContours uses CHAIN_APPROX_SIMPLE (as opposed to SimpleBlobDetector, which uses CHAIN_APPROX_NONE)
- filtering by color actually uses a range of colors
