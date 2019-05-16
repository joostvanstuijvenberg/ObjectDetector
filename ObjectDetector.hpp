/* ============================================================================================== */
/* ObjectDetector.hpp                                                                             */
/*                                                                                                */
/* This file is part of ObjectDetector (github.com/joostvanstuijvenberg/ObjectDetector.git)       */
/*                                                                                                */
/* Joost van Stuijvenberg                                                                         */
/* ============================================================================================== */

#ifndef OBJECTDETECTOR_OBJECTDETECTOR_HPP
#define OBJECTDETECTOR_OBJECTDETECTOR_HPP

#include <utility>
#include <vector>

#include "opencv2/opencv.hpp"

#include "Filter.hpp"
#include "Persistence.hpp"
#include "ThresholdAlgorithm.hpp"

/* ---------------------------------------------------------------------------------------------- */
/* Object detector                                                                                */
/* ---------------------------------------------------------------------------------------------- */
class ObjectDetector {
public:
    inline explicit ObjectDetector(double minDistBetweenObjects = 10.0);
    inline void setThresholdAlgorithm(std::shared_ptr<ThresholdAlgorithm> thresholdAlgorithm) { _thresholdAlgorithm = std::move(
                thresholdAlgorithm); }
    inline double minDistBetweenObjects() { return _minDistBetweenObjects; }
    inline void minDistBetweenObjects(double minDistBetweenObjects) { _minDistBetweenObjects = minDistBetweenObjects; }
    inline void registerFilter(const std::string key, const std::shared_ptr<Filter> filter) { _registeredFilters.emplace(key, filter); }
    inline void addFilter(const std::shared_ptr<Filter> filter) { _filters.emplace_back(filter); }
    inline void clearFilters() { _filters.clear(); }
    std::vector<cv::KeyPoint> detect(const cv::Mat& image);
    inline void read(const cv::FileNode &node);
    inline void write(cv::FileStorage &storage) const;
protected:
    std::vector<Center> findObjects(const cv::Mat &originalImage, const cv::Mat &binaryImage);
private:
    std::map<std::string, std::shared_ptr<ThresholdAlgorithm>> _registeredThresholdAlgorithms;
    std::shared_ptr<ThresholdAlgorithm> _thresholdAlgorithm;
    double _minDistBetweenObjects;
    std::map<std::string, std::shared_ptr<Filter>> _registeredFilters;
    std::vector<std::shared_ptr<Filter>> _filters;
};

ObjectDetector::ObjectDetector(double minDistBetweenObjects)
        : _minDistBetweenObjects(minDistBetweenObjects) {
    _registeredThresholdAlgorithms.emplace("ThresholdFixedAlgorithm", std::make_shared<ThresholdFixedAlgorithm>());
    _registeredThresholdAlgorithms.emplace("ThresholdOtsuAlgorithm", std::make_shared<ThresholdOtsuAlgorithm>());
    _registeredThresholdAlgorithms.emplace("ThresholdRangeAlgorithm", std::make_shared<ThresholdRangeAlgorithm>());

    _registeredFilters.emplace("AreaFilter", std::make_shared<AreaFilter>());
    _registeredFilters.emplace("CircularityFilter", std::make_shared<CircularityFilter>());
    _registeredFilters.emplace("ConvexityFilter", std::make_shared<ConvexityFilter>());
    _registeredFilters.emplace("InertiaFilter", std::make_shared<InertiaFilter>());
    _registeredFilters.emplace("ColorFilter", std::make_shared<ColorFilter>());
    _registeredFilters.emplace("ExtentFilter", std::make_shared<ExtentFilter>());
}

/* ---------------------------------------------------------------------------------------------- */
std::vector<cv::KeyPoint> ObjectDetector::detect(const cv::Mat& image)
{
    assert(image.data != nullptr);
    assert(_thresholdAlgorithm != nullptr);

    // Convert the image to grayscale, when needed.
    cv::Mat gray;
    if (image.channels() == 3 || image.channels() == 4)
        cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    else
        gray = image;
    assert(gray.type() == CV_8UC1);

    _thresholdAlgorithm->setImage(gray);

    std::vector<std::vector<Center>> centers;
    auto binaryImages = _thresholdAlgorithm->binaryImages();
    for (auto binaryImage : binaryImages) {
        auto curCenters = findObjects(gray, binaryImage);

        // Find out the number of occurrences of each object.
        std::vector<std::vector<Center> > newCenters;
        for (auto &curCenter : curCenters) {
            bool isNew = true;
            for (auto &center : centers) {
                double dist = norm(center[center.size() / 2].location - curCenter.location);
                isNew = dist >= _minDistBetweenObjects && dist >= center[center.size() / 2].radius &&
                        dist >= curCenter.radius;
                if (!isNew) {
                    center.push_back(curCenter);
                    size_t k = center.size() - 1;
                    while (k > 0 && center[k].radius < center[k - 1].radius) {
                        center[k] = center[k - 1];
                        k--;
                    }
                    center[k] = curCenter;
                    break;
                }
            }
            if (isNew)
                newCenters.emplace_back(1, curCenter);
        }
        copy(newCenters.begin(), newCenters.end(), back_inserter(centers));
    }

    // Convert the centers that were found into keypoints. Omit centers with less than the specified
    // minimum number of occurrences.
    std::vector<cv::KeyPoint> keypoints;
    for (auto &center : centers) {
        if (center.size() < _thresholdAlgorithm->minRepeatability())
            continue;
        cv::Point2d sumPoint(0, 0);
        double normalizer = 0;
        for (auto &j : center) {
            sumPoint += j.confidence * j.location;
            normalizer += j.confidence;
        }
        sumPoint *= (1. / normalizer);
        cv::KeyPoint kpt(sumPoint, (float) (center[center.size() / 2].radius) * 2.0f);
        keypoints.push_back(kpt);
    }
    return keypoints;
}

/* ---------------------------------------------------------------------------------------------- */
std::vector<Center> ObjectDetector::findObjects(const cv::Mat &originalImage, const cv::Mat &binaryImage)
{
    assert(originalImage.data != nullptr);

    std::vector<Center> centers;

    // Find contours in the binary image using the findContours()-function. Let this function
    // return a list of contours only (no hierarchical data).
    std::vector <std::vector<cv::Point>> contours;
    findContours(binaryImage, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    // Now process all the contours that were found.
    for (auto &contour : contours) {
        Center center;
        center.confidence = 1;
        cv::Moments m = moments(cv::Mat(contour), true); // 2nd parameter specifies image is binary.

        // Skip contours that have no area.
        if (m.m00 == 0.0)
            continue;

        // Process all filters until the first one that filters out the contour.
        bool filtered = false;
        for (const auto &f : _filters)
            if (filtered = f->filter(originalImage, binaryImage, contour, center, m))
                break;
        if (filtered)
            continue;

        // By the time we reach here, the current contour apparently hasn't been filtered out,
        // so compute the location and blob radius and store it as a Center in the centers vector.
        center.location = cv::Point2d(m.m10 / m.m00, m.m01 / m.m00);
        std::vector<double> dists;
        for (const auto &pointIdx : contour) {
            cv::Point2d pt = pointIdx;
            dists.push_back(norm(center.location - pt));
        }
        sort(dists.begin(), dists.end());
        center.radius = (dists[(dists.size() - 1) / 2] + dists[dists.size() / 2]) / 2.;
        centers.push_back(center);
    }
    return centers;
}

void ObjectDetector::read(const cv::FileNode &node) {

    // Threshold algorithm
    auto tas = node[NODE_THRESHOLD_ALGORITHM];
    auto ta = tas.begin();
    auto tan = (*ta).name();
    if (_registeredThresholdAlgorithms.count(tan) == 1) {
        auto t = _registeredThresholdAlgorithms.at(tan);
        t->read(*ta);
        setThresholdAlgorithm(t);
    }
    //TODO: exception when unknown threshold algorithm specified?

    // Minimum distance between Objects
    _minDistBetweenObjects = (double)node[NODE_MIN_DIST_BETWEEN_OBJECTS];

    // Filters
    auto f = node[NODE_FILTERS];
    for (auto fi = f.begin(); fi != f.end(); fi++)
    {
        auto t = (*fi).name();
        if (_registeredFilters.count(t) == 1) {
            auto f = _registeredFilters.at(t);
            f->read(*fi);
            addFilter(f);
        }
        //TODO: exception when unknown filter specified?
    }
}

void ObjectDetector::write(cv::FileStorage &storage) const {
    storage << NODE_THRESHOLD_ALGORITHM << "{";
    _thresholdAlgorithm->write(storage);
    storage << "}";

    storage << NODE_MIN_DIST_BETWEEN_OBJECTS << _minDistBetweenObjects;

    storage << NODE_FILTERS << "{";
    //TODO: filters
    for (const auto &f : _filters)
        f->write(storage);
    storage << "}";
}

#endif //OBJECTDETECTOR_OBJECTDETECTOR_HPP
