/* ============================================================================================== */
/* Filter.hpp                                                                                     */
/*                                                                                                */
/* This file is part of ObjectDetector (github.com/joostvanstuijvenberg/ObjectDetector.git)       */
/*                                                                                                */
/* Joost van Stuijvenberg                                                                         */
/* April 2019                                                                                     */
/* ============================================================================================== */

#ifndef OBJECTDETECTOR_FILTER_H
#define OBJECTDETECTOR_FILTER_H

#include <vector>

#include "opencv2/opencv.hpp"

#include "Center.hpp"

/* ---------------------------------------------------------------------------------------------- */
/* Abstract filter                                                                                */
/* ---------------------------------------------------------------------------------------------- */
class Filter {
public:
    virtual bool filter(const cv::Mat& grayImage, const cv::Mat& binaryImage, const std::vector<cv::Point> &contour, Center &center, const cv::Moments &moments) = 0;
};

/* ---------------------------------------------------------------------------------------------- */
/* Area filter                                                                                    */
/* ---------------------------------------------------------------------------------------------- */
class AreaFilter : public Filter {
public:
    AreaFilter(double min, double max) : _min(min), _max(max) {
        assert (_min <= _max);
    }

    bool filter(const cv::Mat& grayImage, const cv::Mat& binaryImage, const std::vector<cv::Point> &contour, Center &center, const cv::Moments &moments) override {
        return moments.m00 < _min || moments.m00 > _max;
    }

private:
    double _min, _max;
};

/* ---------------------------------------------------------------------------------------------- */
/* Circularity filter                                                                             */
/* ---------------------------------------------------------------------------------------------- */
class CircularityFilter : public Filter {
public:
    CircularityFilter(double min, double max) : _min(min), _max(max) {
        assert (_min <= _max);
    }

    bool filter(const cv::Mat& grayImage, const cv::Mat& binaryImage, const std::vector<cv::Point> &contour, Center &center, const cv::Moments &moments) override {
        double area = moments.m00;
        double perimeter = arcLength(cv::Mat(contour), true);
        double ratio = 4 * CV_PI * area / (perimeter * perimeter);
        return ratio < _min || ratio > _max;
    }

private:
    double _min, _max;
};

/* ---------------------------------------------------------------------------------------------- */
/* Convexity filter                                                                               */
/* ---------------------------------------------------------------------------------------------- */
class ConvexityFilter : public Filter {
public:
    ConvexityFilter(double min, double max) : _min(min), _max(max) {
        assert (_min <= _max);
    }

    bool filter(const cv::Mat& grayImage, const cv::Mat& binaryImage, const std::vector<cv::Point> &contour, Center &center, const cv::Moments &moments) override {
        // If filtering by convexity is requested, skip this contour if the ratio between the contour
        // area and the hull area is not within the specified limits.
        std::vector<cv::Point> hull;
        convexHull(cv::Mat(contour), hull);
        double area = moments.m00;
        double hullArea = contourArea(cv::Mat(hull));
        double ratio = area / hullArea;
        return ratio < _min || ratio > _max;
    }

private:
    double _min, _max;
};

/* ---------------------------------------------------------------------------------------------- */
/* Inertia filter                                                                                 */
/* ---------------------------------------------------------------------------------------------- */
class InertiaFilter : public Filter {
public:
    InertiaFilter(double min, double max) : _min(min), _max(max) {
        assert (_min <= _max);
    }

    bool filter(const cv::Mat& grayImage, const cv::Mat& binaryImage, const std::vector<cv::Point> &contour, Center &center, const cv::Moments &moments) override {
        double denominator = std::sqrt(std::pow(2 * moments.mu11, 2) + std::pow(moments.mu20 - moments.mu02, 2));
        const double eps = 1e-2;
        double ratio;
        if (denominator > eps) {
            double cosmin = (moments.mu20 - moments.mu02) / denominator;
            double sinmin = 2 * moments.mu11 / denominator;
            double cosmax = -cosmin;
            double sinmax = -sinmin;

            double imin =
                    0.5 * (moments.mu20 + moments.mu02) - 0.5 * (moments.mu20 - moments.mu02) * cosmin -
                    moments.mu11 * sinmin;
            double imax =
                    0.5 * (moments.mu20 + moments.mu02) - 0.5 * (moments.mu20 - moments.mu02) * cosmax -
                    moments.mu11 * sinmax;
            ratio = imin / imax;
        } else
            ratio = 1;

        center.confidence = ratio * ratio;
        return ratio < _min || ratio > _max;
    }

private:
    double _min, _max;
};

/* ---------------------------------------------------------------------------------------------- */
/* Color filter                                                                                   */
/* ---------------------------------------------------------------------------------------------- */
class ColorFilter : public Filter {
public:
    explicit ColorFilter(uchar min, uchar max) : _min(min), _max(max) {
        assert (_min <= _max);
    }

    bool filter(const cv::Mat& grayImage, const cv::Mat& binaryImage, const std::vector<cv::Point> &contour, Center &center, const cv::Moments &moments) override {
        // Prevent division by zero, should this contour have no area.
        if (moments.m00 == 0.0)
            return true;

        center.location = cv::Point2d(moments.m10 / moments.m00, moments.m01 / moments.m00);
        auto location = cv::Point2d(moments.m10 / moments.m00, moments.m01 / moments.m00);
        return grayImage.at<uchar>(cvRound(location.y), cvRound(location.x)) < _min || grayImage.at<uchar>(cvRound(location.y), cvRound(location.x)) > _max;
    }

private:
    uchar _min, _max;
};

/* ---------------------------------------------------------------------------------------------- */
/* Extent filter: ratio of contour area to bounding rectangle area                                */
/* ---------------------------------------------------------------------------------------------- */
class ExtentFilter : public Filter {
public:
    ExtentFilter(double min, double max) : _min(min), _max(max) {
        assert (_min <= _max);
    }

    bool filter(const cv::Mat& grayImage, const cv::Mat& binaryImage, const std::vector<cv::Point> &contour, Center &center, const cv::Moments &moments) override {
        double contourArea = moments.m00;
        auto boundingRect = cv::boundingRect(binaryImage);
        double extent = moments.m00 / boundingRect.area();
        return extent < _min || extent > _max;
    }

private:
    double _min, _max;
};

#endif //OBJECTDETECTOR_FILTER_H
