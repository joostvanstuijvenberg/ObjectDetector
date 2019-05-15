/* ============================================================================================== */
/* ThresholdAlgorithm.hpp                                                                         */
/*                                                                                                */
/* This file is part of ObjectDetector (github.com/joostvanstuijvenberg/ObjectDetector.git)       */
/*                                                                                                */
/* Joost van Stuijvenberg                                                                         */
/* ============================================================================================== */

#ifndef OBJECTDETECTOR_THRESHOLDALGORITHM_HPP
#define OBJECTDETECTOR_THRESHOLDALGORITHM_HPP

#include <memory>
#include <sstream>
#include <utility>
#include <vector>

#include "opencv2/opencv.hpp"

#define NODE_THRESHOLD              "threshold"
#define NODE_TYPE                   "type"
#define NODE_MIN                    "min"
#define NODE_MAX                    "max"
#define NODE_STEP                   "step"
#define NODE_MIN_REPEATABLILITY     "minRepeatability"

#define THRESHOLD_ALGORITHM_FIXED   "Fixed"
#define THRESHOLD_ALGORITHM_OTSU    "Otsu"
#define THRESHOLD_ALGORITHM_RANGE   "Range"


/*! /class ThresholdAlgorithm
 *  /brief Abstract threshold algorithm
 *
 *  This class serves as the base of all threshold algorithms. You can subclass your own threshold algorithm from this class.
 */
class ThresholdAlgorithm
{
public:
    inline explicit ThresholdAlgorithm(int minRepeatability = 1) : _minRepeatability(minRepeatability) {}
    inline void setImage(cv::Mat image) { _image = std::move(image); }
    inline int minRepeatability() { return _minRepeatability; }
    virtual std::vector<cv::Mat> binaryImages() = 0;
    virtual void read(const cv::FileNode &node) = 0;
    virtual void write(cv::FileStorage &storage) const = 0;
protected:
    cv::Mat _image;
    int _minRepeatability;
    std::vector<cv::Mat> result;
    void debug(std::vector<cv::Mat>& storage);
};

/*! /brief Debug the thresholding process.
 *
 * This function allows visual inspection of the thresholding process, by showing all the intermediate binary images.
 * @param storage
 */
void ThresholdAlgorithm::debug(std::vector<cv::Mat>& storage)
{
    int w = 0;
    std::vector<std::string> winNames;
    for (const auto &i : storage)
    {
        std::ostringstream os;
        os << "Debug " << w++;
        winNames.emplace_back(os.str());
        cv::namedWindow(os.str());
        cv::imshow(os.str(), i);
    }
    cv::waitKey(0);
    for (const auto &win : winNames)
        cv::destroyWindow(win);
}

/*! /class ThresholdFixedAlgorithm
 *  /brief Implementation of the fixed threshold algorithm
 *
 *  This class implements the fixed threshold algorithm, which uses one single threshold value.
 */
class ThresholdFixedAlgorithm: public ThresholdAlgorithm
{
public:
    inline explicit ThresholdFixedAlgorithm(int threshold = 0)
    : ThresholdAlgorithm(1), _threshold(threshold) {
        assert(_minRepeatability == 1);
    }
    inline std::vector<cv::Mat> binaryImages() override {
        result.clear();
        auto thr = new cv::Mat();
        cv::threshold(_image, *thr, _threshold, 255, cv::THRESH_BINARY);
        result.emplace_back(*thr);
        //debug(result);
        return result;
    }
    inline void read(const cv::FileNode &node) override {
        _threshold = (int)node[NODE_THRESHOLD];
    };
    inline void write(cv::FileStorage &storage) const override {
        storage << NODE_TYPE << THRESHOLD_ALGORITHM_FIXED;
        storage << NODE_THRESHOLD << _threshold;
    };
private:
    int _threshold;
};

/*! /class ThresholdRangeAlgorithm
 *  /brief Implementation of the threshold range algorithm
 *
 *  This class implements the threshold range algorithm, which uses a range of threshold values, specified by
 *  a minimum and maximum threshold value (both inclusive) and a step size.
 */
class ThresholdRangeAlgorithm: public ThresholdAlgorithm
{
public:
    inline explicit ThresholdRangeAlgorithm(int min = 0, int max = 0, int step = 1, int minRepeatability = 0)
    : ThresholdAlgorithm(minRepeatability),_min(min), _max(max), _step(step) {
        assert(_minRepeatability <= (max - min) / step);
    }
    inline std::vector<cv::Mat>  binaryImages() override {
        result.clear();
        for (auto i = _min; i <= _max; i += _step) {
            auto thr = new cv::Mat;
            cv::threshold(_image, *thr, i, 255, cv::THRESH_BINARY);
            result.emplace_back(*thr);
        }
        //debug(result);
        return result;
    }
    inline void read(const cv::FileNode &node) override {
        _min = (int)node[NODE_MIN];
        _max = (int)node[NODE_MAX];
        _step = (int)node[NODE_STEP];
        _minRepeatability = (int)node[NODE_MIN_REPEATABLILITY];
    };
    inline void write(cv::FileStorage &storage) const override {
        storage << NODE_TYPE << THRESHOLD_ALGORITHM_RANGE;
        storage << NODE_MIN << _min;
        storage << NODE_MAX << _max;
        storage << NODE_STEP << _step;
        storage << NODE_MIN_REPEATABLILITY << _minRepeatability;
    };
private:
    int _min, _max, _step;
};

/*! /class ThresholdOtsuAlgorithm
 *  /brief Implementation of Otsu's threshold range algorithm
 *
 *  This class implements Otsu's threshold algorithm.
 */
class ThresholdOtsuAlgorithm: public ThresholdAlgorithm
{
public:
    inline ThresholdOtsuAlgorithm()
    : ThresholdAlgorithm(1) {
        assert(_minRepeatability == 1);
    }
    inline std::vector<cv::Mat>  binaryImages() override {
        result.clear();
        auto thr = new cv::Mat;
        cv::threshold(_image, *thr, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
        result.emplace_back(*thr);
        //debug(result);
        return result;
    }
    inline void read(const cv::FileNode &node) override {};
    inline void write(cv::FileStorage &storage) const override {
        storage << NODE_TYPE << THRESHOLD_ALGORITHM_OTSU;
    };
};

#endif //OBJECTDETECTOR_THRESHOLDALGORITHM_HPP
