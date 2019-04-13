//
// Created by joost on 7-4-19.
//

#ifndef OBJECTDETECTOR_OBJECTDETECTOR_HPP
#define OBJECTDETECTOR_OBJECTDETECTOR_HPP

#include <vector>

#include "opencv2/opencv.hpp"

#include "Filter.hpp"
#include "ThresholdAlgorithm.hpp"

struct Center
{
    cv::Point2d location;
    double radius;
    double confidence;
};

class ObjectDetector {
public:
    void setThresholdAlgorithm(ThresholdAlgorithm* ta) { _thresholdAlgorithm = ta; }
    void addFilter(Filter* filter);
    void detect(cv::Mat image, std::vector<cv::KeyPoint>& keypoints);
protected:
    void findBlobs(cv::Mat image, cv::Mat binaryImage, std::vector<Center> &centers);
private:
    ThresholdAlgorithm* _thresholdAlgorithm;
    std::vector<Filter*> _filters;
};

#endif //OBJECTDETECTOR_OBJECTDETECTOR_HPP
