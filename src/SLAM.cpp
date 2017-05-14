//
// Created by feixue on 17-5-1.
//

#include <SLAM.h>

namespace EpipolarPlaneMatch
{
    SLAM::SLAM(const std::string &strSettingFile) {
        mpTracker = new Tracking(strSettingFile);

        mpMap = new Map();

        mpTracker->SetMap(mpMap);

    }

    SLAM::~SLAM() {
        if (mpTracker)
            delete mpTracker;
    }

    void SLAM::GrabStereoImage(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timeStamp) {
        mpTracker->TrackStereoImage(imLeft, imRight, timeStamp);
    }
}