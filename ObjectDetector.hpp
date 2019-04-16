/* ============================================================================================== */
/* ObjectDetector.hpp                                                                             */
/*                                                                                                */
/* Joost van Stuijvenberg                                                                         */
/* April 2019                                                                                     */
/* ============================================================================================== */

#ifndef OBJECTDETECTOR_OBJECTDETECTOR_HPP
#define OBJECTDETECTOR_OBJECTDETECTOR_HPP

#include <vector>

#include "opencv2/opencv.hpp"

#include "Filter.hpp"
#include "ThresholdAlgorithm.hpp"

/* ---------------------------------------------------------------------------------------------- */
/* Object detector                                                                                */
/* ---------------------------------------------------------------------------------------------- */
class ObjectDetector {
public:
    ObjectDetector(int minRepeatability, double minDistBetweenBlobs)
    : _minRepeatability(minRepeatability), _minDistBetweenBlobs(minDistBetweenBlobs) {}
    void addFilter(std::shared_ptr<Filter> filter);
    void detect(std::shared_ptr<ThresholdAlgorithm> thresholdAlgorithm, cv::Mat& image, std::vector<cv::KeyPoint>& keypoints);
protected:
    void findBlobs(cv::Mat& binaryImage, std::vector<Center> &centers);
private:
    int _minRepeatability;
    double _minDistBetweenBlobs;
    std::vector<std::shared_ptr<Filter>> _filters;
};

#endif //OBJECTDETECTOR_OBJECTDETECTOR_HPP
