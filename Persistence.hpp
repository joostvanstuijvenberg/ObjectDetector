//
// Created by Joost on 16/05/2019.
// This is a facade to OpenCV's XML file storage system.
//

#ifndef OBJECTDETECTOR_PERSISTENCE_HPP
#define OBJECTDETECTOR_PERSISTENCE_HPP

/*! Basic nodes
 *
 */
#define NODE_TYPE                       "type"
#define NODE_MIN                        "min"
#define NODE_MAX                        "max"
#define NODE_STEP                       "step"
#define NODE_MIN_REPEATABLILITY         "minRepeatability"

/*! Structural nodes
 *
 */
#define NODE_FILTERS                    "filters"
#define NODE_THRESHOLD                  "threshold"

/*! Threshold-related nodes
 */
#define NODE_THRESHOLD_ALGORITHM        "thresholdAlgorithm"
#define THRESHOLD_ALGORITHM_FIXED       "Fixed"
#define THRESHOLD_ALGORITHM_OTSU        "Otsu"
#define THRESHOLD_ALGORITHM_RANGE       "Range"
#define NODE_MIN_DIST_BETWEEN_OBJECTS   "minDistBetweenObjects"

#endif //OBJECTDETECTOR_PERSISTENCE_HPP
