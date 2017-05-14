//
// Created by feixue on 17-4-30.
//

#ifndef EPIPOLARPLANEMATCH_TRACKING_H
#define EPIPOLARPLANEMATCH_TRACKING_H

#include <Frame.h>
#include <ORBextractor.h>
#include <ORBmatcher.h>
#include <KeyFrame.h>
#include <Map.h>

namespace EpipolarPlaneMatch
{
    class Map;

    class Tracking
    {
    public:
        enum mTrackingState {
            NO_IMAGES_YET = 0,
            NOT_INITIALIZED = 1,
            SUCCESS = 2,
            LOST = 3
        };

    public:
        Tracking(const std::string& strSettingFile);

        ~Tracking();

    public:
        cv::Mat TrackStereoImage(const cv::Mat& imLeft, const cv::Mat& imRight, double timeStamp);

        void SetMap(Map* pMap);

    protected:
        // Main track function
        void Track();

        // Track strategy
        bool TrackWithMotionModel();

        void UpdateLastFrame();
        bool TrackLocalMap();
        void UpdateLocalMap();
        void SearchLocalPoints();
        void UpdateLocalKeyFrames();
        void UpdateLocalPoints();

        bool TrackEpipolarPlane();

        void CreateNewKeyFrame();

        // process keyframe
        void ProcessCurrentKeyFrame();

        // Stereo initialization
        bool StereoInitialization();

    public:
        mTrackingState mState;

        // Current Frame
        Frame mCurrentFrame;
        Frame mLastFrame;

        cv::Mat mImLeft;
        cv::Mat mImRight;

        // Lists used to recover the full camera trajectory at the end of the execution.
        // Basically we store the reference keyframe for each frame and its relative transformation
        std::list<cv::Mat> mlRelativeFramePoses;
        std::list<KeyFrame*> mlpReferences;
        std::list<double> mlFrameTimes;
        std::list<bool> mlbLost;

    protected:
        ORBextractor* mpORBextractorLeft;
        ORBextractor* mpORBextractorRight;

        cv::Mat mK;
        float mbf;
        float mb;

        float mThDepth;
        bool mbRGB;

        cv::Mat mVelocity;

        KeyFrame* mpReferenceKF;
        std::vector<KeyFrame*> mvpLocalKeyFrames;
        std::vector<MapPoint*> mvpLocalMapPoints;
        Map* mpMap;

    };
}

#endif //EPIPOLARPLANEMATCH_TRACKING_H
