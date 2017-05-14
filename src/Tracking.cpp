//
// Created by feixue on 17-4-30.
//

#include <Tracking.h>
#include <Optimizer.h>
#include <glog/logging.h>

//using namespace cv;
using namespace std;

namespace EpipolarPlaneMatch
{
    Tracking::Tracking(const std::string &strSettingFile) : mState(NO_IMAGES_YET), mpReferenceKF(static_cast<KeyFrame*>(NULL))
    {
        cv::FileStorage fSettings(strSettingFile, cv::FileStorage::READ);
        float fx = fSettings["Camera.fx"];
        float fy = fSettings["Camera.fy"];
        float cx = fSettings["Camera.cx"];
        float cy = fSettings["Camera.cy"];

        cv::Mat K = cv::Mat::eye(3, 3, CV_32F);
        K.at<float>(0, 0) = fx;
        K.at<float>(1, 1) = fy;
        K.at<float>(0, 2) = cx;
        K.at<float>(1, 2) = cy;
        K.copyTo(mK);

        mbf = fSettings["Camera.bf"];
        mb = mbf / fx;

        cout << endl << "Camera Parameters: " << endl;
        cout << "- fx: " << fx << endl;
        cout << "- fy: " << fy << endl;
        cout << "- cx: " << cx << endl;
        cout << "- cy: " << cy << endl;

        int nRGB = fSettings["Camera.RGB"];
        mbRGB = nRGB;

        if (mbRGB)
            cout << "- color order: RGB (ignored if grayscale)" << endl;
        else
            cout << "- color order: BGR (ignored if grayscale)" << endl;

        // Load ORB parameters

        int nFeatures = fSettings["ORBextractor.nFeatures"];
        float fScaleFactor = fSettings["ORBextractor.scaleFactor"];
        int nLevels = fSettings["ORBextractor.nLevels"];
        int fIniThFAST = fSettings["ORBextractor.iniThFAST"];
        int fMinThFAST = fSettings["ORBextractor.minThFAST"];

        mpORBextractorLeft = new ORBextractor(nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);

        mpORBextractorRight = new ORBextractor(nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);


        cout << endl << "ORB Extractor Parameters: " << endl;
        cout << "- Number of Features: " << nFeatures << endl;
        cout << "- Scale Levels: " << nLevels << endl;
        cout << "- Scale Factor: " << fScaleFactor << endl;
        cout << "- Initial Fast Threshold: " << fIniThFAST << endl;
        cout << "- Minimum Fast Threshold: " << fMinThFAST << endl;
    }

    Tracking::~Tracking() {
        if (mpORBextractorLeft)
            delete mpORBextractorLeft;

        if (mpORBextractorRight)
            delete mpORBextractorRight;
    }

    void Tracking::SetMap(Map *pMap) {
        mpMap = pMap;
    }

    cv::Mat Tracking::TrackStereoImage(const cv::Mat &imLeft, const cv::Mat &imRight, double timeStamp) {
        mImLeft = imLeft.clone();
        mImRight = imRight.clone();

        cv::Mat imGrayLeft = imLeft.clone();
        cv::Mat imGrayRight = imRight.clone();

        if (imLeft.channels() == 3)
        {
            cv::cvtColor(imLeft, imGrayLeft, CV_RGB2GRAY);
            cv::cvtColor(imRight, imGrayRight, CV_RGB2GRAY);
        }

        mCurrentFrame = Frame(imGrayLeft, imGrayRight, timeStamp, mpORBextractorLeft, mpORBextractorRight, mK, mbf, mThDepth);

        Track();

        cv::Mat mO;
        cv::cvtColor(mImLeft, mO,cv::COLOR_GRAY2RGB);
        for (int i = 0; i < mCurrentFrame.N; ++i)
        {
            MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
            if (!pMP || mCurrentFrame.mvbOutlier[i]) continue;

            cv::circle(mO, mCurrentFrame.mvKeysLeft[i].pt, 5, cv::Scalar(0, 255, 0), 2);
        }

        cv::imshow("current frame", mO);
        cv::waitKey(10);

        return mCurrentFrame.mTcw.clone();
    }

    void Tracking::Track() {
        bool bOK = false;

        if (mState == NO_IMAGES_YET)
        {
            mState = NOT_INITIALIZED;
            // Initialize the first stereo frame. But the step may be
            bOK = StereoInitialization();

            // Successful initialization
            if (bOK)
                mState = SUCCESS;
        }
        else if (mState == SUCCESS)
        {
            if (mVelocity.empty())
                mVelocity = cv::Mat::eye(4, 4, CV_32F);

            // TrackWithMotionModel
            bOK = TrackWithMotionModel();
            // bOK = TrackEpipolarPlane();
            bOK = TrackLocalMap();

            CreateNewKeyFrame();
            // ProcessCurrentKeyFrame();

            if (!mCurrentFrame.mpReferenceKF)
                mCurrentFrame.mpReferenceKF = mpReferenceKF;

            // Update motion model
            if (bOK) {
                if (!mLastFrame.mTcw.empty()) {
                    cv::Mat LastTwc = cv::Mat::eye(4, 4, CV_32F);
                    mLastFrame.GetRotationInverse().copyTo(LastTwc.rowRange(0, 3).colRange(0, 3));
                    mLastFrame.GetCameraCenter().copyTo(LastTwc.rowRange(0, 3).col(3));
                    mVelocity = mCurrentFrame.mTcw * LastTwc;
                } else
                    mVelocity = cv::Mat::eye(4, 4, CV_32F);
            }
        }


        mLastFrame = Frame(mCurrentFrame);

        if (!mCurrentFrame.mTcw.empty()) {
            cv::Mat Tcr = mCurrentFrame.mTcw * mCurrentFrame.mpReferenceKF->GetPoseInverse();
            mlRelativeFramePoses.push_back(Tcr);
            mlpReferences.push_back(mpReferenceKF);
            mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);
            mlbLost.push_back(mState == LOST);
        } else {
            // This can happen if tracking is lost
            mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());
            mlpReferences.push_back(mlpReferences.back());
            mlFrameTimes.push_back(mlFrameTimes.back());
            mlbLost.push_back(mState == LOST);
        }

    }

    bool Tracking::TrackWithMotionModel() {
        // Do not update the first frame. It's unnecessary!
        if (mLastFrame.mnId > 0)
            UpdateLastFrame();

        mCurrentFrame.SetPose(mLastFrame.mTcw * mVelocity);  // Initialized based on velocity

        ORBmatcher matcher(0.9, true);

        fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(), static_cast<MapPoint *>(NULL));

        int nMatches = matcher.SearchCircleMatchesByProjection(mCurrentFrame, mLastFrame, 7);

        // for debugging

        LOG(INFO) << "Mean reprojection error: " << mCurrentFrame.ComputeMeanReprojectionError();
        LOG(INFO) << "Mean distance to plane: " << mCurrentFrame.ComputeMeanDistanceToPlane();

        int nInliersAfterOptimization = Optimizer::PoseOptimization(&mCurrentFrame); // Optimizer::PoseOptimizationWidthFixedPoints(&mCurrentFrame);//

        LOG(INFO) << "Circle Matches: " << nMatches << " , Inliers after optimization: " << nInliersAfterOptimization;


        for (int i = 0; i < mCurrentFrame.N; ++i)
        {
            MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
            if (pMP && mCurrentFrame.mvbOutlier[i])
                mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
        }

        // Update epipolar planes based on the new camera pose
        mCurrentFrame.UpdatePlanes();

        LOG(INFO) << "Mean reprojection error after optimization: " << mCurrentFrame.ComputeMeanReprojectionError();
        LOG(INFO) << "Mean distance to plane after optimization: " << mCurrentFrame.ComputeMeanDistanceToPlane();

        return nInliersAfterOptimization >= 30;
    }

    void Tracking::UpdateLastFrame() {
        // if (mLastFrame.mTcw.empty()) return;  // This happen when get lost

        KeyFrame *pRef = mLastFrame.mpReferenceKF;
        cv::Mat Tlr = mlRelativeFramePoses.back();

        mLastFrame.SetPose(Tlr * pRef->GetPose());

        // LOG(INFO) << "last frame pose: " << mLastFrame.mTcw;

        int nPoints = 0;
        int nTotalPoints = 0;

        for (int i = 0; i < mLastFrame.N; ++i)
        {
            MapPoint* pMP = mLastFrame.mvpMapPoints[i];
            if (pMP && !mLastFrame.mvbOutlier[i]) continue;
            nTotalPoints++;
            if (mLastFrame.mvDepth[i] > 0)
            {
                cv::Mat X3D = mLastFrame.UnprojectStereo(i);
                MapPoint* pNewMP = new MapPoint(X3D, &mLastFrame, i);
                mLastFrame.mvpMapPoints[i] = pNewMP;
                mLastFrame.mvbOutlier[i] = false;

                nPoints++;
            }
        }

        LOG(INFO) << "Create " << nPoints << " new points " << ", total points: " << nTotalPoints <<  " in frame " << mLastFrame.mnId;
    }

    bool Tracking::TrackLocalMap() {
        // We have an estimation of the camera pose and some map points tracked in the frame.
        // We retrieve the local map and try to find matches to points in the local map.
        UpdateLocalMap();

        SearchLocalPoints();

        // Optimize Pose
        Optimizer::PoseOptimization(&mCurrentFrame);
        int mnMatchesInliers = 0;

        // Update MapPoints Statistics
        for (int i = 0; i < mCurrentFrame.N; i++) {
            if (mCurrentFrame.mvpMapPoints[i]) {
                if (!mCurrentFrame.mvbOutlier[i]) {
                    mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                    if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
                            mnMatchesInliers++;
                    } else
                        mnMatchesInliers++;
                }
            else
                mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);

            }
        // Decide if the tracking was successful
        // More restrictive if there was a relocalization recentl

        LOG(INFO) << "Inliers after tracking local map: " << mnMatchesInliers;
        LOG(INFO) << "Mean reprojection after tracking localmap: " << mCurrentFrame.ComputeMeanReprojectionError();
        // LOG(INFO) << "Mean distance to plane after tracking localmap: " << mCurrentFrame.ComputeMeanDistanceToPlane();

        if (mnMatchesInliers < 30)
            return false;
        else
            return true;
    }

    void Tracking::UpdateLocalMap() {
        // This is for visualization
        // mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

        // Update
        UpdateLocalKeyFrames();
        UpdateLocalPoints();
    }

    void Tracking::SearchLocalPoints() {

        // Do not search map points already matched
        for (std::vector<MapPoint *>::iterator vit = mCurrentFrame.mvpMapPoints.begin(), vend = mCurrentFrame.mvpMapPoints.end();
             vit != vend; vit++) {
            MapPoint *pMP = *vit;
            if (pMP) {
                if (pMP->IsBad()) {
                    *vit = static_cast<MapPoint *>(NULL);
                } else {
                    pMP->IncreaseVisible();
                    pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                    pMP->mbTrackInView = false;
                }
            }
        }

        int nToMatch = 0;

        // Project points in frame and check its visibility
        for (vector<MapPoint *>::iterator vit = mvpLocalMapPoints.begin(), vend = mvpLocalMapPoints.end();
             vit != vend; vit++) {
            MapPoint *pMP = *vit;

            if (pMP->mnLastFrameSeen == mCurrentFrame.mnId)
                continue;
            if (pMP->IsBad())
                continue;
            // Project (this fills MapPoint variables for matching)
            if (mCurrentFrame.IsInFrustum(pMP, 0.5)) {
                pMP->IncreaseVisible();
                nToMatch++;
            }
        }

        if (nToMatch > 0) {
            ORBmatcher matcher(0.8);
            int th = 1;
            // If the camera has been relocalised recently, perform a coarser search
//            if (mCurrentFrame.mnId < mnLastRelocFrameId + 2)
//                th = 5;
            matcher.SearchByProjection(mCurrentFrame, mvpLocalMapPoints, th);
        }
    }

    void Tracking::UpdateLocalKeyFrames() {
        // Each map point vote for the keyframes in which it has been observed
        std::map<KeyFrame *, int> keyframeCounter;
        for (int i = 0; i < mCurrentFrame.N; i++) {
            if (mCurrentFrame.mvpMapPoints[i]) {
                MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];
                if (!pMP->IsBad()) {
                    const map<KeyFrame *, size_t> observations = pMP->GetAllObservations();
                    for (map<KeyFrame *, size_t>::const_iterator it = observations.begin(), itend = observations.end();
                         it != itend; it++)
                        keyframeCounter[it->first]++;
                } else {
                    mCurrentFrame.mvpMapPoints[i] = NULL;
                }
            }
        }

        if (keyframeCounter.empty())
            return;

        int max = 0;
        KeyFrame *pKFmax = static_cast<KeyFrame *>(NULL);

        mvpLocalKeyFrames.clear();
        mvpLocalKeyFrames.reserve(3 * keyframeCounter.size());

        // All keyframes that observe a map point are included in the local map. Also check which keyframe shares most points
        for (map<KeyFrame *, int>::const_iterator it = keyframeCounter.begin(), itEnd = keyframeCounter.end();
             it != itEnd; it++) {
            KeyFrame *pKF = it->first;

            if (pKF->IsBad())
                continue;

            if (it->second > max) {
                max = it->second;
                pKFmax = pKF;
            }

            mvpLocalKeyFrames.push_back(it->first);
            pKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
        }


        // Include also some not-already-included keyframes that are neighbors to already-included keyframes
        for (vector<KeyFrame *>::const_iterator itKF = mvpLocalKeyFrames.begin(), itEndKF = mvpLocalKeyFrames.end();
             itKF != itEndKF; itKF++) {
            // Limit the number of keyframes
            if (mvpLocalKeyFrames.size() > 80)
                break;

            KeyFrame *pKF = *itKF;

            const vector<KeyFrame *> vNeighs = pKF->GetBestCovisibleKeyFrames(10);

            for (vector<KeyFrame *>::const_iterator itNeighKF = vNeighs.begin(), itEndNeighKF = vNeighs.end();
                 itNeighKF != itEndNeighKF; itNeighKF++) {
                KeyFrame *pNeighKF = *itNeighKF;
                if (!pNeighKF->IsBad()) {
                    if (pNeighKF->mnTrackReferenceForFrame != mCurrentFrame.mnId) {
                        mvpLocalKeyFrames.push_back(pNeighKF);
                        pNeighKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                        break;
                    }
                }
            }

//            const set<KeyFrame *> spChilds = pKF->GetChilds();
//            for (set<KeyFrame *>::const_iterator sit = spChilds.begin(), send = spChilds.end(); sit != send; sit++) {
//                KeyFrame *pChildKF = *sit;
//                if (!pChildKF->isBad()) {
//                    if (pChildKF->mnTrackReferenceForFrame != mCurrentFrame.mnId) {
//                        mvpLocalKeyFrames.push_back(pChildKF);
//                        pChildKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
//                        break;
//                    }
//                }
//            }

//            KeyFrame *pParent = pKF->GetParent();
//            if (pParent) {
//                if (pParent->mnTrackReferenceForFrame != mCurrentFrame.mnId) {
//                    mvpLocalKeyFrames.push_back(pParent);
//                    pParent->mnTrackReferenceForFrame = mCurrentFrame.mnId;
//                    break;
//                }
//            }

        }

        if (pKFmax) {
            mpReferenceKF = pKFmax;
            mCurrentFrame.mpReferenceKF = mpReferenceKF;
        }
    }

    void Tracking::UpdateLocalPoints() {
        mvpLocalMapPoints.clear();

        for (vector<KeyFrame *>::const_iterator itKF = mvpLocalKeyFrames.begin(), itEndKF = mvpLocalKeyFrames.end();
             itKF != itEndKF; itKF++) {
            KeyFrame *pKF = *itKF;
            const vector<MapPoint *> vpMPs = pKF->GetMapPointMatches();

            for (vector<MapPoint *>::const_iterator itMP = vpMPs.begin(), itEndMP = vpMPs.end();
                 itMP != itEndMP; itMP++) {
                MapPoint *pMP = *itMP;
                if (!pMP)
                    continue;
//                if (pMP->mType == MapPoint::TEMPORAL) continue;
//
                if (pMP->mnTrackReferenceForFrame == mCurrentFrame.mnId)
                    continue;
                if (!pMP->IsBad()) {
                    mvpLocalMapPoints.push_back(pMP);
                    pMP->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                }
            }
        }
    }

    /**
     * This function is to optimize the camera by using the epipolar plane
     * @return
     */
    bool Tracking::TrackEpipolarPlane() {
        // Optimizer::PoseOptimizationBasedOnPlane(&mCurrentFrame);
        // Optimizer::PoseOptimizationByCombiningPointsAndPlanes(&mCurrentFrame);
        int nInliers = Optimizer::PoseOptimization(&mCurrentFrame);// Optimizer::PoseOptimizationBasedOnPlane(&mCurrentFrame);
        LOG(INFO) << "Number of inliers after plane optimization: " << nInliers;

        mCurrentFrame.UpdatePlanes();

        LOG(INFO) << "Mean reprojection error after tracking plane: " << mCurrentFrame.ComputeMeanReprojectionError();
        LOG(INFO) << "Mean distance to plane after tracking plane: " << mCurrentFrame.ComputeMeanDistanceToPlane();

        return true;
    }

    void Tracking::CreateNewKeyFrame() {
        // Create new MapPoints
        mCurrentFrame.UpdatePoseMatrices();
        mCurrentFrame.UpdatePlanes();

        KeyFrame* pKF = new KeyFrame(mCurrentFrame);
        mpReferenceKF = pKF;
        mCurrentFrame.mpReferenceKF = pKF;

        mpMap->AddKeyFrame(pKF);

        int nPoints = 0;
        for (int i = 0; i < mCurrentFrame.N; ++i)
        {
            if (mCurrentFrame.mvbOutlier[i]) continue;
            MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

            if (pMP)
            {
                pMP->AddObservation(pKF, i);
                pMP->SetReferenceKeyFrame(pKF);
                pKF->AddMapPoint(pMP, i);
                pMP->ComputeDistinctiveDescriptors();
                pMP->UpdateNormalAndDepth();

                mpMap->AddMapPoint(pMP);

                nPoints++;
            }
        }


        LOG(INFO) << "Create new keyframe: " << pKF->mnId << " based on frame: " << mCurrentFrame.mnId
                  << " with new global points: " << nPoints;

        ProcessCurrentKeyFrame();
    }

    bool Tracking::StereoInitialization() {
        // Enough points are necessary
        if (mCurrentFrame.N < 500)
            return false;

        // Set Frame pose to the origin
        mCurrentFrame.SetPose(cv::Mat::eye(4, 4, CV_32F));

        KeyFrame* pNewKF = new KeyFrame(mCurrentFrame);
        mpReferenceKF = pNewKF;
        mCurrentFrame.mpReferenceKF = pNewKF;

        int nNewCreatedPoints = 0;

        for (int i = 0; i < mCurrentFrame.N; ++i)
        {
            const float z = mCurrentFrame.mvDepth[i];
            if (z > 0)
            {
                cv::Mat X3D = mCurrentFrame.UnprojectStereo(i);
                MapPoint* pNewMP = new MapPoint(X3D, &mCurrentFrame, i);
                mCurrentFrame.mvpMapPoints[i] = pNewMP;

                pNewKF->AddMapPoint(pNewMP, i);
                pNewMP->AddObservation(pNewKF, i);
                pNewMP->SetReferenceKeyFrame(pNewKF);
                pNewMP->ComputeDistinctiveDescriptors();
                pNewMP->UpdateNormalAndDepth();

                pNewKF->AddPlane(mCurrentFrame.mvpPlanes[i], i);

                nNewCreatedPoints++;

                mpMap->AddMapPoint(pNewMP);
                mpMap->AddPlane(mCurrentFrame.mvpPlanes[i]);
                mpMap->AddKeyFrame(pNewKF);
            }
        }

        LOG(INFO) << "Mean reprojection error: " << mCurrentFrame.ComputeMeanReprojectionError();
        LOG(INFO) << "Mean distance to plane: " << mCurrentFrame.ComputeMeanDistanceToPlane();

        LOG(INFO) << "Create " << nNewCreatedPoints << " in Map.";

        return true;
    }

    void Tracking::ProcessCurrentKeyFrame() {

        // TODO: update covisibility graph
        mpReferenceKF->UpdateConnections();

        // TODO: local bundle adjustment
        Optimizer::LocalBundleAdjustment(mpReferenceKF);

        LOG(INFO) << "Mean reprojection error in all map: " << mpMap->ComputeMeanReprojectionError();


    }
}
