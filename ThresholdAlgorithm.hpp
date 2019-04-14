/* ---------------------------------------------------------------------------------------------- */

//
// Created by Joost van Stuijvenberg on 7-4-19.
//

#ifndef OBJECTDETECTOR_THRESHOLDALGORITHM_HPP
#define OBJECTDETECTOR_THRESHOLDALGORITHM_HPP

#include <vector>
#include <memory>

#include "opencv2/opencv.hpp"

class ThresholdAlgorithm
{
public:
    void setImage(cv::Mat image) { _image = image; }
    virtual void binaryImages(std::vector<cv::Mat*> &storage) =0;
protected:
    cv::Mat _image;
};

/* ---------------------------------------------------------------------------------------------- */
/* Fixed threshold algorithm                                                                      */
/* ---------------------------------------------------------------------------------------------- */
class FixedThresholdAlgorithm: public ThresholdAlgorithm
{
public:
    FixedThresholdAlgorithm(int threshold): _threshold(threshold) {}
    void binaryImages(std::vector<cv::Mat*> &storage) override {
        cv::Mat* thr = new cv::Mat;
        cv::threshold(_image, *thr, _threshold, 255, cv::THRESH_BINARY);
        storage.emplace_back(thr);
    }
private:
    int _threshold;
};

/* ---------------------------------------------------------------------------------------------- */
/* Threshold range algorithm                                                                      */
/* ---------------------------------------------------------------------------------------------- */
class ThresholdRangeAlgorithm: public ThresholdAlgorithm
{
public:
    ThresholdRangeAlgorithm(int min, int max, int step)
    : _min(min), _max(max), _step(step) {}
    void binaryImages(std::vector<cv::Mat*> &storage) override {
        for (auto i = _min; i <= _max; i += _step) {
            cv::Mat* thr = new cv::Mat;
            cv::threshold(_image, *thr, i, 255, cv::THRESH_BINARY);
            storage.emplace_back(thr);
        }
    }
private:
    int _min, _max, _step;
};

/* ---------------------------------------------------------------------------------------------- */
/* Otsu's threshold algorithm                                                                     */
/* ---------------------------------------------------------------------------------------------- */
class OtsuThresholdAlgorithm: public ThresholdAlgorithm
{
public:
    void binaryImages(std::vector<cv::Mat*> &storage) override {
        cv::Mat* thr = new cv::Mat;
        cv::threshold(_image, *thr, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
        storage.emplace_back(thr);
    }
};

#endif //OBJECTDETECTOR_THRESHOLDALGORITHM_HPP
