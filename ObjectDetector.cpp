/* ============================================================================================== */
/* ObjectDetector.cpp                                                                             */
/*                                                                                                */
/* Joost van Stuijvenberg                                                                         */
/* April 2019                                                                                     */
/* ============================================================================================== */

#include <algorithm>

#include "ObjectDetector.hpp"

/* ---------------------------------------------------------------------------------------------- */
/* addFilter()                                                                                    */
/* ---------------------------------------------------------------------------------------------- */
void ObjectDetector::addFilter(std::shared_ptr<Filter> filter) {
    _filters.emplace_back(filter);
}

/* ---------------------------------------------------------------------------------------------- */
/* detect()                                                                                       */
/* ---------------------------------------------------------------------------------------------- */
std::vector<cv::KeyPoint> ObjectDetector::detect(std::shared_ptr<ThresholdAlgorithm> thresholdAlgorithm, cv::Mat& image)
{
    std::vector<cv::KeyPoint> keypoints;
    assert(image.data != 0);
    assert(thresholdAlgorithm != nullptr);

    cv::Mat gray;
    if (image.channels() == 3 || image.channels() == 4)
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    else
        gray = image;
    assert(gray.type() == CV_8UC1);

    std::vector<cv::Mat *> binaryImages;
    thresholdAlgorithm->setImage(gray);

    std::vector<std::vector<Center>> centers;
    for (auto binaryImage : thresholdAlgorithm->binaryImages()) {
        auto curCenters = findObjects(gray, *binaryImage);

        // Find out the number of occurrences of each object.
        std::vector<std::vector<Center> > newCenters;
        for (auto &curCenter : curCenters) {
            bool isNew = true;
            for (auto &center : centers) {
                double dist = norm(center[center.size() / 2].location - curCenter.location);
                isNew = dist >= _minDistBetweenBlobs && dist >= center[center.size() / 2].radius &&
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
        std::copy(newCenters.begin(), newCenters.end(), std::back_inserter(centers));
    }

    // Skip centers with less than the specified minimum number of occurrences.
    for (auto &center : centers) {
        if (center.size() < thresholdAlgorithm->minRepeatability())
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
/* findObjects()                                                                                  */
/* ---------------------------------------------------------------------------------------------- */
std::vector<Center> ObjectDetector::findObjects(cv::Mat &originalImage, cv::Mat &binaryImage)
{
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

        // SKip contours that have no area.
        if (m.m00 == 0.0)
            continue;

        // Process all filters until the first one that filters out the contour.
        bool filtered = false;
        for (auto f : _filters)
        {
            if (f->filter(originalImage, binaryImage, contour, center, m))
            {
                filtered = true;
                break;
            }
        }

        // If it was filtered out, skip the contour.
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
        std::sort(dists.begin(), dists.end());
        center.radius = (dists[(dists.size() - 1) / 2] + dists[dists.size() / 2]) / 2.;
        centers.push_back(center);
        return centers;
    }
}