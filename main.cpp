#include <iostream>
#include <vector>
#include <memory>

#include "opencv2/opencv.hpp"

#include "ObjectDetector.hpp"
#include "ThresholdAlgorithm.hpp"

int main(int argc, char** argv) {

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

    cv::namedWindow("Original");
    cv::imshow("Original", image);

    ThresholdAlgorithm* ta = new ThresholdRangeAlgorithm(40, 120, 10);
    ObjectDetector od(ta);
    Filter* f1 = new AreaFilter(100, 5000);
    od.addFilter(f1);

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