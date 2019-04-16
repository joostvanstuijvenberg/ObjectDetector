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

/* ---------------------------------------------------------------------------------------------- */
/* showResults() - utility function to show the results of object detection                       */
/* ---------------------------------------------------------------------------------------------- */
void showResults(const std::string &title, const cv::Mat& image, const std::vector<cv::KeyPoint>& keypoints)
{
    static int x = 200;
    static int y = 200;

    cv::Mat result;
    cv::drawKeypoints(image, keypoints, result, cv::Scalar(0, 0, 255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    cv::namedWindow(title);
    cv::moveWindow(title, x, y);
    cv::imshow(title, result);

    x += 100;
    y += 100;
}

/* ---------------------------------------------------------------------------------------------- */
/* main()                                                                                         */
/* ---------------------------------------------------------------------------------------------- */
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

    // Use the threshold range algorithm to find objects using a range of thresholds (min, max, step)
    // and with a minimum repeatability.
    auto tra = std::make_shared<ThresholdRangeAlgorithm>(40, 150, 10, 3);
    // Use Otsu's threshold algorithm.
    auto ota = std::make_shared<OtsuThresholdAlgorithm>();

    ObjectDetector od(MIN_DIST_BETWEEN_BLOBS);

    // We'll use an area filter first.
    od.addFilter(std::make_shared<AreaFilter>(4000, 50000));
    od.detect(tra, image, keypoints);
    showResults("Filtering by area, using a threshold range", image, keypoints);

    // Now we add a circularity filter.
    od.addFilter(std::make_shared<CircularityFilter>(0.75, 1.0));
    od.detect(tra, image, keypoints);
    showResults("Filtering by area and circularity, using a threshold range", image, keypoints);

    // Filtering by both area and inertia.
    od.resetFilters();
    od.addFilter(std::make_shared<AreaFilter>(4000, 15000));
    od.addFilter(std::make_shared<InertiaFilter>(0.05, 0.75));
    od.detect(tra, image, keypoints);
    showResults("Filtering by area and inertia, using a threshold range", image, keypoints);

    // Now just select Otsu's threshold algorithm.
    od.detect(ota, image, keypoints);
    showResults("Filtering by area and inertia, using Otsu's threshold algorithm", image, keypoints);

    // Press <Esc> to quit this demo.
    while(cv::waitKey(40) != 27);
    return 0;
}