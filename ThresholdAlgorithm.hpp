/* ============================================================================================== */
/* ThresholdAlgorithm.hpp                                                                         */
/*                                                                                                */
/* Joost van Stuijvenberg                                                                         */
/* April 2019                                                                                     */
/* ============================================================================================== */

#ifndef OBJECTDETECTOR_THRESHOLDALGORITHM_HPP
#define OBJECTDETECTOR_THRESHOLDALGORITHM_HPP

#include <vector>
#include <memory>
#include <sstream>

#include "opencv2/opencv.hpp"

/* ---------------------------------------------------------------------------------------------- */
/* Abstract threshold algorithm                                                                   */
/* ---------------------------------------------------------------------------------------------- */
class ThresholdAlgorithm
{
public:
    void setImage(cv::Mat image) { _image = image; }
    virtual void getBinaryImages(std::vector<cv::Mat *> &storage) = 0;
protected:
    cv::Mat _image;
    void debug(std::vector<cv::Mat*>& storage)
    {
        int w = 0;
        std::vector<std::string> winNames;
        for (auto i : storage)
        {
            std::ostringstream os;
            os << "Debug " << w++;
            winNames.emplace_back(os.str());
            cv::namedWindow(os.str());
            cv::imshow(os.str(), *i);
        }
        cv::waitKey(0);
        for (auto w : winNames)
            cv::destroyWindow(w);
    }
};

/* ---------------------------------------------------------------------------------------------- */
/* Fixed threshold algorithm                                                                      */
/* ---------------------------------------------------------------------------------------------- */
class FixedThresholdAlgorithm: public ThresholdAlgorithm
{
public:
    FixedThresholdAlgorithm(int threshold): _threshold(threshold) {}
    void getBinaryImages(std::vector<cv::Mat *> &storage) override {
        cv::Mat* thr = new cv::Mat;
        cv::threshold(_image, *thr, _threshold, 255, cv::THRESH_BINARY);
        storage.emplace_back(thr);
        debug(storage);
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
    void getBinaryImages(std::vector<cv::Mat *> &storage) override {
        for (auto i = _min; i <= _max; i += _step) {
            cv::Mat* thr = new cv::Mat;
            cv::threshold(_image, *thr, i, 255, cv::THRESH_BINARY);
            storage.emplace_back(thr);
        }
        debug(storage);
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
    void getBinaryImages(std::vector<cv::Mat *> &storage) override {
        cv::Mat* thr = new cv::Mat;
        cv::threshold(_image, *thr, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
        storage.emplace_back(thr);
        debug(storage);
    }
};

#endif //OBJECTDETECTOR_THRESHOLDALGORITHM_HPP
