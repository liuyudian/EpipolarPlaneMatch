//
// Created by feixue on 17-4-29.
//

#ifndef FRAME_H
#define FRAME_H

#include <ORBextractor.h>
#include <Plane.h>
#include <MapPoint.h>

#include <opencv2/opencv.hpp>

namespace EpipolarPlaneMatch {

    class MapPoint;
    class Plane;
    class KeyFrame;

#define FRAME_GRID_ROWS 48
#define FRAME_GRID_COLS 64

    class Frame {
    public:
        Frame();

        // Copy constructor.
        Frame(const Frame &frame);

        // Constructor for stereo cameras.
        Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timeStamp, ORBextractor *extractorLeft,
              ORBextractor *extractorRight, cv::Mat &K, const float &bf, const float &thDepth);

        // Extract ORB on the image. 0 for left image and 1 for right image.
        void ExtractORB(int flag, const cv::Mat &im);


        // Set the camera pose.
        void SetPose(cv::Mat Tcw);

        // Computes rotation, translation and camera center matrices from the camera pose.
        void UpdatePoseMatrices();

        // Update plane
        void UpdatePlanes();

        // Returns the camera center.
        inline cv::Mat GetCameraCenter() {
            return mOw.clone();
        }

        // Returns inverse of rotation
        inline cv::Mat GetRotationInverse() {
            return mRwc.clone();
        }

        // Check if a MapPoint is in the frustum of the camera
        // and fill variables of the MapPoint to be used by the tracking
        // bool isInFrustum(MapPoint *pMP, float viewingCosLimit);

        // Compute the cell of a keypoint (return false if outside the grid)
        bool PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY);

        bool IsInFrustum(MapPoint* pMP, float view);

        std::vector <size_t> GetFeaturesInArea(const float &x, const float &y, const float &r, const int minLevel = -1,
                                          const int maxLevel = -1) const;

        // Search a match for each keypoint in the left image to a keypoint in the right image.
        // If there is a match, depth is computed and the right coordinate associated to the left keypoint is stored.
        void ComputeStereoMatches();

        // Backprojects a keypoint (if stereo/depth info available) into 3D world coordinates.
        cv::Mat UnprojectStereo(const int &i);

        float ComputeMeanReprojectionError();

        float ComputeMeanDistanceToPlane();



    public:

        // Feature extractor. The right is used only in the stereo case.
        ORBextractor *mpORBextractorLeft, *mpORBextractorRight;

        // Frame timestamp.
        double mTimeStamp;

        // Calibration matrix and OpenCV distortion parameters.
        cv::Mat mK;
        static float fx;
        static float fy;
        static float cx;
        static float cy;
        static float invfx;
        static float invfy;

        static float mnMinX, mnMinY;
        static float mnMaxX, mnMaxY;

        // Stereo baseline multiplied by fx.
        float mbf;

        // Stereo baseline in meters.
        float mb;

        // Threshold close/far points. Close points are inserted from 1 view.
        // Far points are inserted as in the monocular case from 2 views.
        float mThDepth;

        // Number of KeyPoints.
        int N;

        // Associated 3D points and planes
        std::vector<MapPoint*> mvpMapPoints;
        std::vector<Plane*> mvpPlanes;

        // Vector of keypoints (original for visualization) and undistorted (actually used by the system).
        // In the stereo case, mvKeysUn is redundant as images must be rectified.
        // In the RGB-D case, RGB images can be distorted.
        std::vector<cv::KeyPoint> mvKeysLeft, mvKeysRight;
        std::vector<int> mnMatches;

        // Corresponding stereo coordinate and depth for each keypoint.
        // "Monocular" keypoints have a negative value.
        std::vector<float> mvuRight;
        std::vector<float> mvDepth;

        // ORB descriptor, each row associated to a keypoint.
        cv::Mat mDescriptorsLeft, mDescriptorsRight;

        // Flag to identify outlier associations.
        std::vector<bool> mvbOutlier;

        // Keypoints are assigned to cells in a grid to reduce matching complexity when projecting MapPoints.
        static float mfGridElementWidthInv;
        static float mfGridElementHeightInv;
        std::vector<std::size_t> mGrid[FRAME_GRID_COLS][FRAME_GRID_ROWS];

        // Camera pose.
        cv::Mat mTcw;

        KeyFrame* mpReferenceKF;
        // Current and Next Frame id.
        static long unsigned int nNextId;
        long unsigned int mnId;

        // Scale pyramid info.
        int mnScaleLevels;
        float mfScaleFactor;
        float mfLogScaleFactor;
        std::vector<float> mvScaleFactors;
        std::vector<float> mvInvScaleFactors;
        std::vector<float> mvLevelSigma2;
        std::vector<float> mvInvLevelSigma2;

        static bool mbInitialComputations;

    private:
        // Assign keypoints to the grid for speed up feature matching (called in the constructor).
        void AssignFeaturesToGrid();

        // Initialize epipolar plane
        void InitializeEpipolarPlane();

        // Rotation, translation and camera center
        cv::Mat mRcw;
        cv::Mat mtcw;
        cv::Mat mRwc;
        cv::Mat mOw; //==mtwc
    };

}// namespace ORB_SLAM

#endif // FRAME_H
