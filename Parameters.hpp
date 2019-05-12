//
// Created by joost on 21-4-19.
//

#ifndef OBJECTDETECTOR_PARAMETERS_HPP
#define OBJECTDETECTOR_PARAMETERS_HPP

#include <string>
#include <vector>

#include "ObjectDetector.hpp"

#define MIN_DIST_BETWEEN_BLOBS  "minDistBetweenBlobs"

struct Parameters {

    ThresholdAlgorithm* thresholdAlgorithm = nullptr;
    double minDistBetweenBlobs = 0.0;
    std::vector<Filter*> filters;

    void read(const cv::FileNode &node)
    {
        std::string t = (std::string)node["type"];
        if (t == "Fixed")
            thresholdAlgorithm = new ThresholdFixedAlgorithm;
        if (t == "Otsu")
            thresholdAlgorithm = new ThresholdOtsuAlgorithm;
        if (t == "Range")
            thresholdAlgorithm = new ThresholdRangeAlgorithm;
        thresholdAlgorithm->read(node);
        minDistBetweenBlobs = (double)node[MIN_DIST_BETWEEN_BLOBS];
    }

    void write(cv::FileStorage &storage) const
    {
        thresholdAlgorithm->write(storage);
        storage << MIN_DIST_BETWEEN_BLOBS << minDistBetweenBlobs;
    }

};

#endif //OBJECTDETECTOR_PARAMETERS_HPP
