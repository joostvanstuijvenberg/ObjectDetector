/* ============================================================================================== */
/* main.cpp                                                                                       */
/*                                                                                                */
/* This file demonstrates the use of ObjectDetector; an enhanced version of OpenCV's Simple-      */
/* BlobDetector. ObjectDetector allows different thresholding algorithms to be used and various   */
/* filters to be applied. This can be changed run-time and filtering parameters can be specified  */
/* programmatically.                                                                              */
/*                                                                                                */
/* Please note that this file just contains a demo, and you can throw it away and implement the   */
/* ObjectDetector class in your own application.                                                  */
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
#include "Parameters.hpp"

/* ---------------------------------------------------------------------------------------------- */
/* showWindow() - utility function to show the results of object detection                       */
/* ---------------------------------------------------------------------------------------------- */
void showWindow(const std::string &title, const cv::Mat &image, const std::vector<cv::KeyPoint> *keypoints) {
    static auto x = 100;
    static auto y = 50;

    cv::Mat result;
    if (keypoints != nullptr)
        cv::drawKeypoints(image, *keypoints, result, cv::Scalar(0, 0, 255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    else
        result = image;

    cv::namedWindow(title);
    cv::moveWindow(title, x, y);
    cv::imshow(title, result);

    x += 100;
    y += 50;
}

/* ---------------------------------------------------------------------------------------------- */
/* main()                                                                                         */
/* ---------------------------------------------------------------------------------------------- */
int main(int argc, char **argv) {

#define GENERATE_XML
#ifdef GENERATE_XML
    // Use this as a convenience to generate the xml parameter file.
    cv::FileStorage temp("parameters.xml", cv::FileStorage::WRITE);
    Parameters x;
    x.thresholdAlgorithm = new ThresholdRangeAlgorithm(40, 150, 10, 3);
    //p.thresholdAlgorithm = new ThresholdOtsuAlgorithm;
    x.minDistBetweenBlobs = 10.0;
    x.write(temp);
    temp.release();
#endif

    // See if a filename was specified as the first parameter and try to open and show it.
    if (argc != 2) {
        std::cout << "Usage: ObjectDetector {filename}" << std::endl;
        exit(EXIT_FAILURE);
    }

    cv::Mat image = cv::imread(argv[1]);
    if (!image.data) {
        std::cout << "Could not load file " << argv[1] << std::endl;
        exit(EXIT_FAILURE);
    }

    showWindow("Original", image, nullptr);

    // Create the threshold range algorithm to find objects using a range of thresholds (min, max, step)
    // and with a specified minimum repeatability.
    auto tra = std::make_shared<ThresholdRangeAlgorithm>(40, 150, 10, 3);

    // Create Otsu's threshold algorithm.
    auto ota = std::make_shared<ThresholdOtsuAlgorithm>();

    // Create a fixed threshold algorithm.
    auto fta = std::make_shared<ThresholdFixedAlgorithm>(100);

    // Create an object detector. Use 10.0 as the minimum distance between BLOBs.
    ObjectDetector od(10.0);

    // We'll use an area filter first.
    od.addFilter(std::make_shared<AreaFilter>(4000, 50000));
    auto keypoints = od.detect(tra, image);
    showWindow("Area: 4000 - 50000, threshold algorithm: range", image, &keypoints);

    // Now we add a circularity filter.
    od.addFilter(std::make_shared<CircularityFilter>(0.75, 1.0));
    keypoints = od.detect(tra, image);
    showWindow("Area: 4000 - 50000, circularity: 0.75 - 1.0, threshold algorithm: range 40 - 150, step 10", image,
               &keypoints);

    // Filtering by both area and inertia.
    od.resetFilters();
    od.addFilter(std::make_shared<AreaFilter>(4000, 15000));
    od.addFilter(std::make_shared<InertiaFilter>(0.05, 0.75));
    keypoints = od.detect(tra, image);
    showWindow("Area: 4000 - 50000, inertia: 0.05 - 0.75, threshold algorithm: range 40 - 150, step 10", image,
               &keypoints);

    // Now just select Otsu's threshold algorithm.
    keypoints = od.detect(ota, image);
    showWindow("Area: 4000 - 50000, inertia: 0.05 - 0.75, threshold algorithm: Otsu", image, &keypoints);

    // Now we will show all objects that have a medium gray to white 'color'.
    od.resetFilters();
    od.addFilter(std::make_shared<AreaFilter>(1000, 50000));
    od.addFilter(std::make_shared<ColorFilter>(140, 160));
    keypoints = od.detect(tra, image);
    showWindow("Area: 1000 - 50000, gray value: 140 - 160, threshold algorithm: range 40 - 150, step 10", image,
               &keypoints);

    // Now let's select the least convex object from the image.
    od.resetFilters();
    od.addFilter(std::make_shared<AreaFilter>(1000, 5000));
    od.addFilter(std::make_shared<ConvexityFilter>(0.0, 0.6));
    keypoints = od.detect(ota, image);
    showWindow("Area: 1000 - 5000, convexity: 0.0 - 0.6, threshold algorithm: Otsu", image, &keypoints);

    // Start over with an extent filter.
    od.resetFilters();
    od.addFilter(std::make_shared<AreaFilter>(5000, 50000));
    od.addFilter(std::make_shared<ExtentFilter>(0.02, 0.04));
    keypoints = od.detect(fta, image);
    showWindow("Area: 5000 - 50000, extent ratio: 0.02 - 0.04, threshold algorithm: fixed 100", image, &keypoints);

    // Now construct an object detector using parameters.xml
    cv::FileStorage storage("parameters.xml", cv::FileStorage::READ);
    cv::FileNode node = storage["opencv_storage"];
    Parameters p;
    p.read(node);
    ObjectDetector o2(p.minDistBetweenBlobs);

    // Press <Esc> to quit this demo.
    while (cv::waitKey(500) != 27);
    return 0;
}