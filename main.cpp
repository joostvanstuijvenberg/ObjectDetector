/* ============================================================================================== */
/* main.cpp                                                                                       */
/*                                                                                                */
/* This file demonstrates the use of ObjectDetector; an enhanced version of OpenCV's Simple-      */
/* BlobDetector. ObjectDetector allows different thresholding algorithms to be used and various   */
/* filters to be applied. This can be changed run-time and filtering parameters can be specified  */
/* programmatically.                                                                              */
/*                                                                                                */
/* Joost van Stuijvenberg                                                                         */
/* April 2019                                                                                     */
/* ============================================================================================== */

#include <iostream>
#include <vector>
#include <memory>

#include "opencv2/opencv.hpp"

#include "ObjectDetector.hpp"
#include "ThresholdAlgorithm.hpp"

int MIN_REPEATABILITY = 3;
double MIN_DIST_BETWEEN_BLOBS = 10.0;

int main(int argc, char** argv) {

    // See if a filename was specified as the first parameter and try to open and show it.
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

    std::vector<cv::KeyPoint> keypoints;

    std::string Original {"Original"};
    cv::namedWindow(Original);
    cv::moveWindow(Original, 100, 100);
    cv::imshow(Original, image);

    // Use the threshold range algorithm to find objects using a range of thresholds (min, max, step) and the
    // repeatability.
    auto tra = std::make_shared<ThresholdRangeAlgorithm>(40, 120, 10, 3);

    ObjectDetector od(MIN_DIST_BETWEEN_BLOBS);

    // We'll use an area filter first.
    std::string Results1 {"Results filter 1: by area"};
    od.addFilter(std::make_shared<AreaFilter>(4000, 50000));
    od.detect(tra, image, keypoints);
    cv::Mat results1;
    cv::drawKeypoints(image, keypoints, results1, cv::Scalar(0, 0, 255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    cv::namedWindow(Results1);
    cv::moveWindow(Results1, 300, 300);
    cv::imshow(Results1, results1);

    // Now we add a circularity filter.
    std::string Results2 {"Results filter 2: by circularity"};
    od.addFilter(std::make_shared<CircularityFilter>(0.75, 1.0));
    od.detect(tra, image, keypoints);
    cv::Mat results2;
    cv::drawKeypoints(image, keypoints, results2, cv::Scalar(0, 0, 255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    cv::namedWindow(Results2);
    cv::moveWindow(Results2, 500, 500);
    cv::imshow(Results2, results2);

    cv::waitKey(0);
    return 0;
}