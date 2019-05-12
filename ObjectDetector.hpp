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
    explicit ObjectDetector(double minDistBetweenBlobs = 0.0)
    : _minDistBetweenBlobs(minDistBetweenBlobs) {}
    void setThresholdAlgorithm(std::shared_ptr<ThresholdAlgorithm> thresholdAlgorithm) { _thresholdAlgorithm = thresholdAlgorithm; }
    double getMinDistBetweenBlobs() { return _minDistBetweenBlobs; }
    void setMinDistBetweenBlobs(double minDistBetweenBlobs) { _minDistBetweenBlobs = minDistBetweenBlobs; }
    void addFilter(std::shared_ptr<Filter> filter) { _filters.emplace_back(filter); }
    void resetFilters() { _filters.clear(); }
    std::vector<cv::KeyPoint> detect(cv::Mat& image);
    void read(const cv::FileNode &node)
    {
        auto ta = node[NODE_THRESHOLD_ALGORITHM];
        auto t = (std::string)ta[NODE_TYPE];
        if (t == THRESHOLD_ALGORITHM_FIXED)
            _thresholdAlgorithm = std::make_shared<ThresholdFixedAlgorithm>();
        if (t == THRESHOLD_ALGORITHM_OTSU)
            _thresholdAlgorithm = std::make_shared<ThresholdOtsuAlgorithm>();
        if (t == THRESHOLD_ALGORITHM_RANGE)
            _thresholdAlgorithm = std::make_shared<ThresholdRangeAlgorithm>();
        _thresholdAlgorithm->read(node);

        _minDistBetweenBlobs = (double)node[NODE_MIN_DIST_BETWEEN_BLOBS];

        auto f = node["filters"];
        for (auto fi = f.begin(); fi != f.end(); fi++)
        {
            auto t = (*fi).string();
            std::cout << t << std::endl;
        }
        //TODO: filters
    }
    void write(cv::FileStorage &storage) const
    {
        storage << NODE_THRESHOLD_ALGORITHM << "{";
        _thresholdAlgorithm->write(storage);
        storage << "}";

        storage << NODE_MIN_DIST_BETWEEN_BLOBS << _minDistBetweenBlobs;

        storage << NODE_FILTERS << "{";
        //TODO: filters
        for (auto f : _filters)
            f->write(storage);
        storage << "}";
    }

protected:
    std::vector<Center> findObjects(cv::Mat &originalImage, cv::Mat &binaryImage);
private:
    std::shared_ptr<ThresholdAlgorithm> _thresholdAlgorithm;
    double _minDistBetweenBlobs;
    std::vector<std::shared_ptr<Filter>> _filters;
};

#endif //OBJECTDETECTOR_OBJECTDETECTOR_HPP
