/* ============================================================================================== */
/* Filter.hpp                                                                                     */
/*                                                                                                */
/* This file is part of ObjectDetector (github.com/joostvanstuijvenberg/ObjectDetector.git)       */
/*                                                                                                */
/* Joost van Stuijvenberg                                                                         */
/* ============================================================================================== */

#ifndef OBJECTDETECTOR_FILTER_H
#define OBJECTDETECTOR_FILTER_H

#include <vector>

#include "opencv2/opencv.hpp"

#include "Persistence.hpp"

/* ---------------------------------------------------------------------------------------------- */
/* Center data structure                                                                          */
/* ---------------------------------------------------------------------------------------------- */
struct Center
{
    cv::Point2d location;
    double radius;
    double confidence;
};

/* ---------------------------------------------------------------------------------------------- */
/* Abstract filter                                                                                */
/* ---------------------------------------------------------------------------------------------- */
class Filter {
public:
    virtual bool filter(const cv::Mat& grayImage, const cv::Mat& binaryImage, const std::vector<cv::Point> &contour, Center &center, const cv::Moments &moments) = 0;
    virtual void read(const cv::FileNode &node) = 0;
    virtual void write(cv::FileStorage &storage) const = 0;
};

/* ---------------------------------------------------------------------------------------------- */
/* Area filter                                                                                    */
/* ---------------------------------------------------------------------------------------------- */
class AreaFilter : public Filter {
public:
    inline explicit AreaFilter(double min = 0, double max = 0) : _min(min), _max(max) {
        assert (_min <= _max);
    }
    inline bool filter(const cv::Mat& grayImage, const cv::Mat& binaryImage, const std::vector<cv::Point> &contour, Center &center, const cv::Moments &moments) override {
        return moments.m00 < _min || moments.m00 > _max;
    }
    inline double minArea() const { return _min; }
    inline void minArea(double min) { _min = min; }
    inline double maxArea() const { return _max; }
    inline void maxArea(double max) { _max = max; }
    inline void read(const cv::FileNode &node) override {
        _min = (double)node[NODE_MIN];
        _max = (double)node[NODE_MAX];
    }
    inline void write(cv::FileStorage &storage) const override {
        storage << "AreaFilter" << "{";
        storage << NODE_MIN << _min;
        storage << NODE_MAX << _max;
        storage << "}";
    };
private:
    double _min, _max;
};

/* ---------------------------------------------------------------------------------------------- */
/* Circularity filter                                                                             */
/* ---------------------------------------------------------------------------------------------- */
class CircularityFilter : public Filter {
public:
    inline explicit CircularityFilter(double min = 0, double max = 0) : _min(min), _max(max) {
        assert (_min <= _max);
    }
    inline bool filter(const cv::Mat& grayImage, const cv::Mat& binaryImage, const std::vector<cv::Point> &contour, Center &center, const cv::Moments &moments) override {
        double area = moments.m00;
        double perimeter = arcLength(cv::Mat(contour), true);
        double ratio = 4 * CV_PI * area / (perimeter * perimeter);
        return ratio < _min || ratio > _max;
    }
    inline double minCircularity() const { return _min; }
    inline void minCircularity(double min) { _min = min; }
    inline double maxCircularity() const { return _max; }
    inline void maxCircularity(double max) { _max = max; }
    inline void read(const cv::FileNode &node) override {
        _min = (double)node[NODE_MIN];
        _max = (double)node[NODE_MAX];
    };
    inline void write(cv::FileStorage &storage) const override {
        storage << "CircularityFilter" << "{";
        storage << NODE_MIN << _min;
        storage << NODE_MAX << _max;
        storage << "}";
    };
private:
    double _min, _max;
};

/* ---------------------------------------------------------------------------------------------- */
/* Convexity filter                                                                               */
/* ---------------------------------------------------------------------------------------------- */
class ConvexityFilter : public Filter {
public:
    inline explicit ConvexityFilter(double min = 0, double max = 0) : _min(min), _max(max) {
        assert (_min <= _max);
    }
    inline bool filter(const cv::Mat& grayImage, const cv::Mat& binaryImage, const std::vector<cv::Point> &contour, Center &center, const cv::Moments &moments) override {
        // If filtering by convexity is requested, skip this contour if the ratio between the contour
        // area and the hull area is not within the specified limits.
        std::vector<cv::Point> hull;
        convexHull(cv::Mat(contour), hull);
        double area = moments.m00;
        double hullArea = contourArea(cv::Mat(hull));
        double ratio = area / hullArea;
        return ratio < _min || ratio > _max;
    }
    inline double minConvexity() const { return _min; }
    inline void minConvexity(double min) { _min = min; }
    inline double maxConvexity() const { return _max; }
    inline void maxConvexity(double max) { _max = max; }
    inline void read(const cv::FileNode &node) override {
        _min = (double)node[NODE_MIN];
        _max = (double)node[NODE_MAX];
    };
    inline void write(cv::FileStorage &storage) const override {
        storage << "ConvexityFilter" << "{";
        storage << NODE_MIN << _min;
        storage << NODE_MAX << _max;
        storage << "}";
    };
private:
    double _min, _max;
};

/* ---------------------------------------------------------------------------------------------- */
/* Inertia filter                                                                                 */
/* ---------------------------------------------------------------------------------------------- */
class InertiaFilter : public Filter {
public:
    inline explicit InertiaFilter(double min = 0, double max = 0) : _min(min), _max(max) {
        assert (_min <= _max);
    }
    inline bool filter(const cv::Mat& grayImage, const cv::Mat& binaryImage, const std::vector<cv::Point> &contour, Center &center, const cv::Moments &moments) override {
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
    inline double minInertia() const { return _min; }
    inline void minInertia(double min) { _min = min; }
    inline double maxInertia() const { return _max; }
    inline void maxInertia(double max) { _max = max; }
    inline void read(const cv::FileNode &node) override {
        _min = (double)node[NODE_MIN];
        _max = (double)node[NODE_MAX];
    };
    inline void write(cv::FileStorage &storage) const override {
        storage << "InertiaFilter" << "{";
        storage << NODE_MIN << _min;
        storage << NODE_MAX << _max;
        storage << "}";
    };
private:
    double _min, _max;
};

/* ---------------------------------------------------------------------------------------------- */
/* Color filter                                                                                   */
/* ---------------------------------------------------------------------------------------------- */
class ColorFilter : public Filter {
public:
    inline explicit ColorFilter(uchar min = 0, uchar max = 0) : _min(min), _max(max) {
        assert (_min <= _max);
    }
    inline bool filter(const cv::Mat& grayImage, const cv::Mat& binaryImage, const std::vector<cv::Point> &contour, Center &center, const cv::Moments &moments) override {
        // Prevent division by zero, should this contour have no area.
        if (moments.m00 == 0.0)
            return true;
        center.location = cv::Point2d(moments.m10 / moments.m00, moments.m01 / moments.m00);
        auto location = cv::Point2d(moments.m10 / moments.m00, moments.m01 / moments.m00);
        return grayImage.at<uchar>(cvRound(location.y), cvRound(location.x)) < _min || grayImage.at<uchar>(cvRound(location.y), cvRound(location.x)) > _max;
    }
    inline uchar minColor() const { return _min; }
    inline void minColor(uchar min) { _min = min; }
    inline uchar maxColor() const { return _max; }
    inline void maxColor(uchar max) { _max = max; }
    inline void read(const cv::FileNode &node) override {
        _min = (uchar)(int)node[NODE_MIN];
        _max = (uchar)(int)node[NODE_MAX];
    };
    inline void write(cv::FileStorage &storage) const override {
        storage << "ColorFilter" << "{";
        storage << NODE_MIN << _min;
        storage << NODE_MAX << _max;
        storage << "}";
    };
private:
    uchar _min, _max;
};

/* ---------------------------------------------------------------------------------------------- */
/* Extent filter: ratio of contour area to bounding rectangle area                                */
/* ---------------------------------------------------------------------------------------------- */
class ExtentFilter : public Filter {
public:
    inline explicit ExtentFilter(double min = 0, double max = 0) : _min(min), _max(max) {
        assert (_min <= _max);
    }
    inline bool filter(const cv::Mat& grayImage, const cv::Mat& binaryImage, const std::vector<cv::Point> &contour, Center &center, const cv::Moments &moments) override {
        auto contourArea = moments.m00;
        auto boundingRect = cv::boundingRect(binaryImage);
        auto extent = moments.m00 / boundingRect.area();
        return extent < _min || extent > _max;
    }
    inline double minExtent() const { return _min; }
    inline void minExtent(double min) { _min = min; }
    inline double maxExtent() const { return _max; }
    inline void maxExtent(double max) { _max = max; }
    inline void read(const cv::FileNode &node) override {
        _min = (double)node[NODE_MIN];
        _max = (double)node[NODE_MAX];
    };
    inline void write(cv::FileStorage &storage) const override {
        storage << "ExtentFilter" << "{";
        storage << NODE_MIN << _min;
        storage << NODE_MAX << _max;
        storage << "}";
    };
private:
    double _min, _max;
};

#endif //OBJECTDETECTOR_FILTER_H