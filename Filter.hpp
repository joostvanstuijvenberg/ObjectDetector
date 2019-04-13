//
// Created by joost on 10-4-19.
//

#ifndef OBJECTDETECTOR_FILTER_H
#define OBJECTDETECTOR_FILTER_H

#include <vector>

#include "opencv2/opencv.hpp"

class Filter {
public:
    virtual bool filter(const cv::Mat &image, const std::vector<cv::Point> &contour, const cv::Moments &moments) = 0;

private:
};

/* Area filter */
class AreaFilter : public Filter {
public:
    AreaFilter(int min, int max) : _min(min), _max(max) {}

    bool filter(const cv::Mat &image, const std::vector<cv::Point> &contour, const cv::Moments &moments) override {
        return moments.m00 < _min || moments.m00 > _max;
    }

private:
    int _min, _max;
};

/* Circularity filter */
class CircularityFilter : public Filter {
public:
    CircularityFilter(int min, int max) : _min(min), _max(max) {}

    bool filter(const cv::Mat &image, const std::vector<cv::Point> &contour, const cv::Moments &moments) override {
        double area = moments.m00;
        double perimeter = arcLength(cv::Mat(contour), true);
        double ratio = 4 * CV_PI * area / (perimeter * perimeter);
        return ratio < _min || ratio > _max;
    }

private:
    int _min, _max;
};

/* Convexity filter */
class ConvexityFilter : public Filter {
public:
    ConvexityFilter(int min, int max) : _min(min), _max(max) {}

    bool filter(const cv::Mat &image, const std::vector<cv::Point> &contour, const cv::Moments &moments) override {
        // If filtering by convexity is requested, skip this contour if the ratio between the contour
        // area and the hull area is not within the specified limits.
        std::vector<cv::Point> hull;
        convexHull(cv::Mat(contour), hull);
        double area = contourArea(cv::Mat(contour));
        double hullArea = contourArea(cv::Mat(hull));
        double ratio = area / hullArea;
        return ratio < _min || ratio > _max;
    }

private:
    int _min, _max;
};

/* Inertia filter */
class InertiaFilter : public Filter {
public:
    InertiaFilter(int min, int max) : _min(min), _max(max) {}

    bool filter(const cv::Mat &image, const std::vector<cv::Point> &contour, const cv::Moments &moments) override {
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

//TODO:        center.confidence = ratio * ratio;

        return ratio < _min || ratio > _max;
    }

private:
    int _min, _max;
};

/* Color filter */
class ColorFilter : public Filter {
public:
    ColorFilter(uchar color) : _color(color) {}

    bool filter(const cv::Mat &image, const std::vector<cv::Point> &contour, const cv::Moments &moments) override {
        // Prevent division by zero, should this contour have no area.
        if (moments.m00 == 0.0)
            return true;

        //TODO: center.location = cv::Point2d(moms.m10 / moms.m00, moms.m01 / moms.m00);
        cv::Point2d location = cv::Point2d(moments.m10 / moments.m00, moments.m01 / moments.m00);

        //return !(binaryImage.at<uchar>(cvRound(center.location.y), cvRound(center.location.x)) != _color);
        return !(image.at<uchar>(cvRound(location.y), cvRound(location.x)) != _color);
    }

private:
    uchar _color;
};

/* Bending energy filter */
class BendingEnergyFilter : public Filter {
public:
    BendingEnergyFilter(int min, int max) : _min(min), _max(max) {}

    bool filter(const cv::Mat &image, const std::vector<cv::Point> &contour, const cv::Moments &moments) override {
        return moments.m00 < _min || moments.m00 > _max;
    }

private:
    int _min, _max;
};

#endif //OBJECTDETECTOR_FILTER_H
