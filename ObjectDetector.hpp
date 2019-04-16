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
    explicit ObjectDetector(ThresholdAlgorithm* thresholdAlgorithm, int minRepeatability, double minDistBetweenBlobs)
    : _thresholdAlgorithm(thresholdAlgorithm), _minRepeatability(minRepeatability), _minDistBetweenBlobs(minDistBetweenBlobs) {}
    void addFilter(Filter* filter);
    void detect(cv::Mat image, std::vector<cv::KeyPoint>& keypoints);
protected:
    void findBlobs(cv::Mat binaryImage, std::vector<Center> &centers);
private:
    ThresholdAlgorithm* _thresholdAlgorithm;
    int _minRepeatability;
    double _minDistBetweenBlobs;
    std::vector<Filter*> _filters;
};

#endif //OBJECTDETECTOR_OBJECTDETECTOR_HPP
