//
// Created by feixue on 17-4-30.
//

#ifndef EPIPOLARPLANEMATCH_SLAM_H
#define EPIPOLARPLANEMATCH_SLAM_H

#include <Frame.h>
#include <MapPoint.h>
#include <Map.h>
#include <Tracking.h>

#include <opencv2/opencv.hpp>
#include <string>

namespace EpipolarPlaneMatch
{
    class Map;

    class SLAM
    {
    public:
        SLAM(const std::string& strSettingFile);
        ~SLAM();

    public:
        void GrabStereoImage(const cv::Mat& imLeft, const cv::Mat& imRight, const double& timeStamp = 1.0);


    protected:
        Tracking* mpTracker;

        Map* mpMap;

    };
}

#endif //EPIPOLARPLANEMATCH_SLAM_H
