#include <iostream>
#include <vector>
#include <memory>

#include "opencv2/opencv.hpp"

#include "ObjectDetector.hpp"
#include "ThresholdAlgorithm.hpp"

int main(int argc, char** argv) {

    // See if a filename was specified as the first parameter and try to open it.
    if (argc != 2)
    {
        std::cout << "Usage: ObjectDetector {filename}" << std::endl;
        exit(EXIT_FAILURE);
    }

    cv::Mat image = cv::imread(argv[1]);
    if (! image.data)
    {
        std::cout << "Could not load file " << argv[1] << std::endl;
        exit(EXIT_FAILURE);
    }

    // Show the original image.
    cv::namedWindow("Original");
    cv::imshow("Original", image);

    // Use the threshold range algorithm to find objects using different thresholds. Then create an area filter.
    ThresholdAlgorithm* ta = new ThresholdRangeAlgorithm(40, 120, 10);
    Filter* f1 = new AreaFilter(100, 500);
    ObjectDetector od(ta);
    od.addFilter(f1);

    // Detect objects and return their centers as keypoints. Then show these keypoints in a second image window.
    std::vector<cv::KeyPoint> keypoints;
    od.detect(image, keypoints);
    std::cout << keypoints.size() << " keypoints found." << std::endl;
    cv::Mat results;
    cv::drawKeypoints(image, keypoints, results, cv::Scalar(0, 0, 255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    cv::namedWindow("Results");
    cv::moveWindow("Results", 600, 400);
    cv::imshow("Results", results);

    cv::waitKey(0);
    delete ta;
    delete f1;
    return 0;
}