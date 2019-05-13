# ObjectDetector
An enhanced version of OpenCV's SimpleBlobDetector

## Benefits over SimpleBlobDetector
- features multiple threshold algorithms, including Otsu's
- user defined filter classes
- findContours uses CHAIN_APPROX_SIMPLE (as opposed to SimpleBlobDetector, which uses CHAIN_APPROX_NONE)
- filtering by color actually uses a range of colors