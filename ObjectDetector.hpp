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
    explicit ObjectDetector(double minDistBetweenBlobs)
    : _minDistBetweenBlobs(minDistBetweenBlobs) {}
    void addFilter(std::shared_ptr<Filter> filter);
    void resetFilters() { _filters.clear(); }
    std::vector<cv::KeyPoint> detect(std::shared_ptr<ThresholdAlgorithm> thresholdAlgorithm, cv::Mat& image);
protected:
    std::vector<Center> findObjects(cv::Mat &originalImage, cv::Mat &binaryImage);
private:
    double _minDistBetweenBlobs;
    std::vector<std::shared_ptr<Filter>> _filters;
};

#endif //OBJECTDETECTOR_OBJECTDETECTOR_HPP
