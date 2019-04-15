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
    explicit ObjectDetector(ThresholdAlgorithm* thresholdAlgorithm) : _thresholdAlgorithm(thresholdAlgorithm) {}
    void addFilter(Filter* filter);
    void detect(cv::Mat image, std::vector<cv::KeyPoint>& keypoints);
protected:
    void findBlobs(cv::Mat image, cv::Mat binaryImage, std::vector<Center> &centers);
private:
    ThresholdAlgorithm* _thresholdAlgorithm;
    std::vector<Filter*> _filters;
};

#endif //OBJECTDETECTOR_OBJECTDETECTOR_HPP
