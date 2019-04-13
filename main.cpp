#include <iostream>
#include <vector>
#include <memory>

#include "opencv2/opencv.hpp"

#include "ObjectDetector.hpp"
#include "ThresholdAlgorithm.hpp"

int main() {
    cv::Mat image = cv::imread("/home/joost/Data/Projects/OpenCV/ObjectDetector/testbeeld.jpg");
    cv::namedWindow("Original");
    cv::imshow("Original", image);

    std::vector<cv::KeyPoint> keypoints;
    ObjectDetector od;

    ThresholdAlgorithm* ta = new ThresholdRangeAlgorithm(40, 120, 10);
    od.setThresholdAlgorithm(ta);

    Filter* f1 = new AreaFilter(100, 5000);
    od.addFilter(f1);

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