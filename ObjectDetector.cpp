/* ============================================================================================== */
/* ObjectDetector.cpp                                                                             */
/*                                                                                                */
/* Joost van Stuijvenberg                                                                         */
/* April 2019                                                                                     */
/* ============================================================================================== */

#include "ObjectDetector.hpp"

double MIN_REPEATABILITY = 3.0;
double MIN_DIST_BETWEEN_BLOBS = 10.0;

/* ---------------------------------------------------------------------------------------------- */
/* addFilter()                                                                                    */
/* ---------------------------------------------------------------------------------------------- */
void ObjectDetector::addFilter(Filter *filter) {
    _filters.emplace_back(filter);
}

/* ---------------------------------------------------------------------------------------------- */
/* detect()                                                                                       */
/* ---------------------------------------------------------------------------------------------- */
void ObjectDetector::detect(cv::Mat image, std::vector<cv::KeyPoint> &keypoints) {

    // An image with actual data must be passed to this function.
    assert(image.data != 0);

    // A threshold algorithm must be set *before* calling detect().
    assert(_thresholdAlgorithm != nullptr);

    keypoints.clear();

    cv::Mat gray;
    if (image.channels() == 3 || image.channels() == 4)
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    else
        gray = image;

    // ObjectDetector only supports 8-bit image depth.
    assert(gray.type() == CV_8UC1);

    std::vector<cv::Mat *> binaryImages;
    _thresholdAlgorithm->setImage(gray);
    _thresholdAlgorithm->binaryImages(binaryImages);

    std::vector<std::vector<Center> > centers;
    for (auto binarizedImage : binaryImages) {
        std::vector<Center> curCenters;
        findBlobs(gray, *binarizedImage, curCenters);
        std::vector<std::vector<Center> > newCenters;
        for (size_t i = 0; i < curCenters.size(); i++) {
            bool isNew = true;
            for (size_t j = 0; j < centers.size(); j++) {
                double dist = norm(centers[j][centers[j].size() / 2].location - curCenters[i].location);
                isNew = dist >= MIN_DIST_BETWEEN_BLOBS && dist >= centers[j][centers[j].size() / 2].radius &&
                        dist >= curCenters[i].radius;
                if (!isNew) {
                    centers[j].push_back(curCenters[i]);

                    size_t k = centers[j].size() - 1;
                    while (k > 0 && centers[j][k].radius < centers[j][k - 1].radius) {
                        centers[j][k] = centers[j][k - 1];
                        k--;
                    }
                    centers[j][k] = curCenters[i];

                    break;
                }
            }
            if (isNew)
                newCenters.emplace_back(1, curCenters[i]);
        }
        std::copy(newCenters.begin(), newCenters.end(), std::back_inserter(centers));
    }

    for (auto &center : centers) {
        if (center.size() < MIN_REPEATABILITY)
            continue;
        cv::Point2d sumPoint(0, 0);
        double normalizer = 0;
        for (size_t j = 0; j < center.size(); j++) {
            sumPoint += center[j].confidence * center[j].location;
            normalizer += center[j].confidence;
        }
        sumPoint *= (1. / normalizer);
        cv::KeyPoint kpt(sumPoint, (float) (center[center.size() / 2].radius) * 2.0f);
        keypoints.push_back(kpt);
    }
}

/* ---------------------------------------------------------------------------------------------- */
/* findBlobs                                                                                      */
// PRE  : _binaryImage contains a valid binary image
// PRE  : centers is a reference to a vector of Center instances
// POST : centers contains Center objects for each blob that was found
/* ---------------------------------------------------------------------------------------------- */
void ObjectDetector::findBlobs(cv::Mat image, cv::Mat binaryImage, std::vector<Center> &centers) {

    // Find contours in the binary image using the findContours()-function. Let this function
    // return a list of contours only (no hierarchical data).
    centers.clear();
    std::vector <std::vector<cv::Point>> contours;
    findContours(binaryImage, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

    // Now process all the contours that were found.
    for (auto &contour : contours) {
        Center center;
        center.confidence = 1;
        cv::Moments moms = moments(cv::Mat(contour));

        bool filtered = false;
        for (Filter* f : _filters)
        {
            if (f->filter(image, contour, moms))
            {
                filtered = true;
                break;
            }
        }

        if (filtered)
            continue;

        // By the time we reach here, the current contour apparently hasn't been filtered out,
        // so we compute the blob radius and store it as a Center in the centers vector.
        std::vector<double> dists;
        for (auto &pointIdx : contour) {
            cv::Point2d pt = pointIdx;
            dists.push_back(norm(center.location - pt));
        }
        std::sort(dists.begin(), dists.end());
        center.radius = (dists[(dists.size() - 1) / 2] + dists[dists.size() / 2]) / 2.;
        centers.push_back(center);
    }
}