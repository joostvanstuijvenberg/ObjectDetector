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

#include <vector>

#include "opencv2/opencv.hpp"

#include "Filter.hpp"
#include "ThresholdAlgorithm.hpp"

/* ---------------------------------------------------------------------------------------------- */
/* Object detector                                                                                */
/* ---------------------------------------------------------------------------------------------- */
class ObjectDetector {
public:
    explicit ObjectDetector(double minDistBetweenBlobs = 0.0)
    : _minDistBetweenBlobs(minDistBetweenBlobs) {}
    void setThresholdAlgorithm(std::shared_ptr<ThresholdAlgorithm> thresholdAlgorithm) { _thresholdAlgorithm = thresholdAlgorithm; }
    void addFilter(std::shared_ptr<Filter> filter);
    void resetFilters() { _filters.clear(); }
    std::vector<cv::KeyPoint> detect(cv::Mat& image);
    void read(const cv::FileNode &node)
    {
        auto t = (std::string)node["type"];
        if (t == "Fixed")
            _thresholdAlgorithm = std::make_shared<ThresholdFixedAlgorithm>();
        if (t == "Otsu")
            _thresholdAlgorithm = std::make_shared<ThresholdOtsuAlgorithm>();
        if (t == "Range")
            _thresholdAlgorithm = std::make_shared<ThresholdRangeAlgorithm>();
        _thresholdAlgorithm->read(node);
        _minDistBetweenBlobs = (double)node["minDistBetweenBlobs"];
    }
    void write(cv::FileStorage &storage) const
    {
        _thresholdAlgorithm->write(storage);
        storage << "minDistBetweenBlobs" << _minDistBetweenBlobs;
    }

protected:
    std::vector<Center> findObjects(cv::Mat &originalImage, cv::Mat &binaryImage);
private:
    std::shared_ptr<ThresholdAlgorithm> _thresholdAlgorithm;
    double _minDistBetweenBlobs;
    std::vector<std::shared_ptr<Filter>> _filters;
};

#endif //OBJECTDETECTOR_OBJECTDETECTOR_HPP
