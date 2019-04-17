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
    explicit ThresholdAlgorithm(int minRepeatability = 1) : _minRepeatability(minRepeatability) {}
    void setImage(cv::Mat image) { _image = image; }
    virtual std::vector<cv::Mat> binaryImages() = 0;
    int minRepeatability() { return _minRepeatability; }
protected:
    cv::Mat _image;
    int _minRepeatability;
    std::vector<cv::Mat> result;
    void debug(std::vector<cv::Mat>& storage)
    {
        int w = 0;
        std::vector<std::string> winNames;
        for (auto i : storage)
        {
            std::ostringstream os;
            os << "Debug " << w++;
            winNames.emplace_back(os.str());
            cv::namedWindow(os.str());
            cv::imshow(os.str(), i);
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
    explicit FixedThresholdAlgorithm(int threshold)
    : ThresholdAlgorithm(), _threshold(threshold) {
        assert(_minRepeatability == 1);
    }
    std::vector<cv::Mat> binaryImages() override {
        result.clear();
        auto thr = new cv::Mat();
        cv::threshold(_image, *thr, _threshold, 255, cv::THRESH_BINARY);
        result.emplace_back(*thr);
        //debug(result);
        return result;
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
    ThresholdRangeAlgorithm(int min, int max, int step, int minRepeatability)
    : ThresholdAlgorithm(minRepeatability),_min(min), _max(max), _step(step) {
        assert(_minRepeatability <= (max - min) / step);
    }
    std::vector<cv::Mat>  binaryImages() override {
        result.clear();
        for (auto i = _min; i <= _max; i += _step) {
            auto* thr = new cv::Mat;
            cv::threshold(_image, *thr, i, 255, cv::THRESH_BINARY);
            result.emplace_back(*thr);
        }
        debug(result);
        return result;
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
    OtsuThresholdAlgorithm()
    : ThresholdAlgorithm() {
        assert(_minRepeatability == 1);
    }
    std::vector<cv::Mat>  binaryImages() override {
        result.clear();
        auto* thr = new cv::Mat;
        cv::threshold(_image, *thr, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
        result.emplace_back(*thr);
        //debug(storage);
        return result;
    }
};

#endif //OBJECTDETECTOR_THRESHOLDALGORITHM_HPP
