/* ============================================================================================== */
/* Center.hpp                                                                                     */
/*                                                                                                */
/* Joost van Stuijvenberg                                                                         */
/* April 2019                                                                                     */
/* ============================================================================================== */

#ifndef OBJECTDETECTOR_CENTER_HPP
#define OBJECTDETECTOR_CENTER_HPP

#include "opencv2/opencv.hpp"

/* ---------------------------------------------------------------------------------------------- */
/* Center data structure                                                                          */
/* ---------------------------------------------------------------------------------------------- */
struct Center
{
    cv::Point2d location;
    double radius;
    double confidence;
};

#endif //OBJECTDETECTOR_CENTER_HPP
