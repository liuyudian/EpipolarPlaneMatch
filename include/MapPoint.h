//
// Created by feixue on 17-4-30.
//

#ifndef EPIPOLARPLANEMATCH_MAPPOINT_H
#define EPIPOLARPLANEMATCH_MAPPOINT_H

#include <Plane.h>
#include <Frame.h>
#include <KeyFrame.h>

#include <opencv2/opencv.hpp>

#include <mutex>

namespace EpipolarPlaneMatch
{
    class Map;
    class Frame;
    class KeyFrame;

    class MapPoint
    {
    public:
        MapPoint();

        MapPoint(const cv::Mat& Pos, Frame* pFrame, const int& idxF);

        ~MapPoint();

    public:
        void SetWorldPos(cv::Mat Pos);

        cv::Mat GetWorldPos();

        cv::Mat GetNormal();

        void AddObservation(KeyFrame* pKF, const size_t& idx);

        void EraseObservation(KeyFrame* pKF);

        void SetReferenceKeyFrame(KeyFrame* pKF);

        std::map<KeyFrame* ,size_t> GetAllObservations();

        void ComputeDistinctiveDescriptors();

        cv::Mat GetDescriptor();

        void UpdateNormalAndDepth();

        int GetIndexInKeyFrame(KeyFrame* pKF);

        // Is the this point bad or not
        void SetBad();
        bool IsBad() const;

        void IncreaseVisible(const int& n = 1);
        void IncreaseFound(const int& n = 1);

        int Observations();

        float GetMaxDistanceInvariance();
        float GetMinDistanceInvariance();
        int PredictScale(const float& dist, Frame* pF);
        int PredictScale(const float &currentDist, KeyFrame *pKF);

    public:
        unsigned long mnId;
        static unsigned long nNextId;
        int nObs;
        long int mnFirstFrame;

        unsigned long mnBALocalKF;
        unsigned long mnTrackReferenceForFrame;
        unsigned long mnLastFrameSeen;

        // Tracking parameters
        bool mbTrackInView;

        float mTrackProjX;
        float mTrackProjXR;
        float mTrackProjY;
        int mnTrackScaleLevel;
        float mTrackViewCos;

    protected:
        // Mean viewing direction
        cv::Mat mNormalVector;

        // Best descriptor to fast matching
        cv::Mat mDescriptor;

        // Position in world coordinate
        cv::Mat mWorldPos;

        // Reference KeyFrame
        KeyFrame* mpRefKF;

        float mfMaxDistance;
        float mfMinDistance;

        // Sign of the point
        bool bBad;

        // observed parameters
        unsigned long mnFound;
        unsigned long mnVisible;

        // Map* mpMap;

        std::map<KeyFrame*, size_t> mObservations;

        std::mutex mMutexPos;
        std::mutex mMutexFeatures;


    };
}

#endif //EPIPOLARPLANEMATCH_MAPPOINT_H
