//
// Created by feixue on 17-4-30.
//

#ifndef EPIPOLARPLANEMATCH_KEYFRAME_H
#define EPIPOLARPLANEMATCH_KEYFRAME_H

#include <Frame.h>
#include <MapPoint.h>

namespace EpipolarPlaneMatch
{
    class MapPoint;
    class Frame;

    class KeyFrame
    {
    public:
        KeyFrame(Frame& frame);

        ~KeyFrame();

        // Pose functions
        void SetPose(const cv::Mat &Tcw);
        cv::Mat GetPose();
        cv::Mat GetPoseInverse();
        cv::Mat GetCameraCenter();
        cv::Mat GetStereoCenter();
        cv::Mat GetRotation();
        cv::Mat GetTranslation();

        // MapPoint observations
        void AddMapPoint(MapPoint* pMP, const size_t& idx);
        void EraseMapPointMatch(MapPoint* pMP);
        void EraseMapPointMatch(const int& idx);
        std::vector<MapPoint*> GetValidMapPoints();
        std::vector<MapPoint*> GetMapPointMatches();
        MapPoint* GetMapPoint(const size_t& idx);

        // Epipolar plane observations
        void AddPlane(Plane* pPlane, const size_t& idx);
        std::vector<Plane*> GetValidPlanes();
        std::vector<Plane*> GetPlaneMatches();
        Plane* GetPlane(const size_t& idx);

        // Covisibility graph
        void AddConnection(KeyFrame* pKF, const int weight);
        void UpdateConnections();
        void UpdateBestCovisibility();

        std::vector<KeyFrame*> GetVectorCovisibleKeyFrames();

        std::vector<KeyFrame*> GetBestCovisibleKeyFrames(const int & n);

        // Image
        bool IsInImage(const float &x, const float &y) const;

        bool IsBad() const;
        void SetBad();

        // KeyPoint functions
        std::vector<size_t> GetFeaturesInArea(const float &x, const float  &y, const float  &r) const;
        cv::Mat UnprojectStereo(int i);

        static bool weightComp( int a, int b){
            return a>b;
        }

        static bool lId(KeyFrame* pKF1, KeyFrame* pKF2){
            return pKF1->mnId<pKF2->mnId;
        }

    public:
        static long unsigned int nNextId;
        long unsigned int mnId;
        const long unsigned int mnFrameId;

        const double mTimeStamp;

        // Grid (to speed up feature matching)
        const int mnGridCols;
        const int mnGridRows;
        const float mfGridElementWidthInv;
        const float mfGridElementHeightInv;

        // Calibration parameters
        const float fx, fy, cx, cy, invfx, invfy, mbf, mb, mThDepth;
        const float mHalfBaseline;

        // Number of KeyPoints
        const int N;

        // KeyPoints, stereo coordinate and descriptors (all associated by an index)
        const std::vector<cv::KeyPoint> mvKeys;
        const std::vector<float> mvuRight; // negative value for monocular points
        const std::vector<float> mvDepth; // negative value for monocular points
        const cv::Mat mDescriptors;

        // Scale
        const int mnScaleLevels;
        const float mfScaleFactor;
        const float mfLogScaleFactor;
        const std::vector<float> mvScaleFactors;
        const std::vector<float> mvLevelSigma2;
        const std::vector<float> mvInvLevelSigma2;

        // Image bounds and calibration
        const int mnMinX;
        const int mnMinY;
        const int mnMaxX;
        const int mnMaxY;
        const cv::Mat mK;


        // Tag for local bundle adjustment
        unsigned long mnBALocalKF;
        unsigned long mnBAFixedKF;
        unsigned long mnTrackReferenceForFrame;

    protected:
        // SE3 Pose and camera center
        cv::Mat Tcw;
        cv::Mat Twc;
        cv::Mat Ow;

        cv::Mat Cw; // Stereo middel point. Only for visualization

        bool bBad;

        // MapPoints associated to keypoints
        std::vector<MapPoint*> mvpMapPoints;

        // Planes associated to MapPoints
        std::vector<Plane*> mvpPlanes;

        // Grid over the image to speed up feature matching
        std::vector< std::vector <std::vector<size_t> > > mGrid;

        std::map<KeyFrame*,int> mConnectedKeyFrameWeights;
        std::vector<KeyFrame*> mvpOrderedConnectedKeyFrames;
        std::vector<int> mvOrderedWeights;
    };
}

#endif //EPIPOLARPLANEMATCH_KEYFRAME_H
