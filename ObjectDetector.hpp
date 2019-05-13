/* ============================================================================================== */
/* ObjectDetector.hpp                                                                             */
/*                                                                                                */
/* This file is part of ObjectDetector (github.com/joostvanstuijvenberg/ObjectDetector.git)       */
/*                                                                                                */
/* Joost van Stuijvenberg                                                                         */
/* April 2019                                                                                     */
/* ============================================================================================== */

#ifndef OBJECTDETECTOR_OBJECTDETECTOR_HPP
#define OBJECTDETECTOR_OBJECTDETECTOR_HPP

#define NODE_MIN_DIST_BETWEEN_BLOBS     "minDistBetweenBlobs"
#define NODE_THRESHOLD_ALGORITHM        "thresholdAlgorithm"
#define NODE_FILTERS                    "filters"

#include <vector>

#include "opencv2/opencv.hpp"

#include "Filter.hpp"
#include "ThresholdAlgorithm.hpp"

/* ---------------------------------------------------------------------------------------------- */
/* Object detector                                                                                */
/* ---------------------------------------------------------------------------------------------- */
class ObjectDetector {
public:
    explicit ObjectDetector(double minDistBetweenBlobs = 0.0);
    void setThresholdAlgorithm(std::shared_ptr<ThresholdAlgorithm> thresholdAlgorithm) { _thresholdAlgorithm = thresholdAlgorithm; }
    double getMinDistBetweenBlobs() { return _minDistBetweenBlobs; }
    void setMinDistBetweenBlobs(double minDistBetweenBlobs) { _minDistBetweenBlobs = minDistBetweenBlobs; }
    void registerFilter(std::string key, std::shared_ptr<Filter> filter) { _registeredFilters.emplace(key, filter); }
    void addFilter(std::shared_ptr<Filter> filter) { _filters.emplace_back(filter); }
    void resetFilters() { _filters.clear(); }
    std::vector<cv::KeyPoint> detect(cv::Mat& image);
    void read(const cv::FileNode &node);
    void write(cv::FileStorage &storage) const;
protected:
    std::vector<Center> findObjects(cv::Mat &originalImage, cv::Mat &binaryImage);
private:
    std::shared_ptr<ThresholdAlgorithm> _thresholdAlgorithm;
    double _minDistBetweenBlobs;
    std::map<std::string, std::shared_ptr<Filter>> _registeredFilters;
    std::vector<std::shared_ptr<Filter>> _filters;
};

#endif //OBJECTDETECTOR_OBJECTDETECTOR_HPP
