//
// Created by feixue on 17-4-29.
//

#include <Frame.h>
#include <ORBmatcher.h>

#include <thread>
#include <glog/logging.h>

using namespace std;

namespace EpipolarPlaneMatch {

    long unsigned int Frame::nNextId = 0;
    bool Frame::mbInitialComputations = true;
    float Frame::cx, Frame::cy, Frame::fx, Frame::fy, Frame::invfx, Frame::invfy;
    float Frame::mnMinX, Frame::mnMinY, Frame::mnMaxX, Frame::mnMaxY;
    float Frame::mfGridElementWidthInv, Frame::mfGridElementHeightInv;

    Frame::Frame() {

    }

    Frame::Frame(const Frame &frame) : mnId(frame.mnId), mpORBextractorLeft(frame.mpORBextractorLeft),
                                       mpORBextractorRight(mpORBextractorRight),
                                       mTimeStamp(frame.mTimeStamp), mK(frame.mK.clone()), mbf(frame.mbf), mb(frame.mb),
                                       mThDepth(frame.mThDepth),
                                       N(frame.N), mvKeysLeft(frame.mvKeysLeft), mvKeysRight(frame.mvKeysRight),
                                       mvuRight(frame.mvuRight),mpReferenceKF(frame.mpReferenceKF),
                                       mnMatches(frame.mnMatches), mvDepth(frame.mvDepth),
                                       mDescriptorsLeft(frame.mDescriptorsLeft.clone()),
                                       mDescriptorsRight(frame.mDescriptorsRight.clone()), mvpMapPoints(frame.mvpMapPoints),
                                       mvpPlanes(frame.mvpPlanes), mvbOutlier(frame.mvbOutlier),mnScaleLevels(frame.mnScaleLevels),
                                       mfScaleFactor(frame.mfScaleFactor), mfLogScaleFactor(frame.mfLogScaleFactor),
                                       mvScaleFactors(frame.mvScaleFactors), mvInvScaleFactors(frame.mvInvScaleFactors),
                                       mvLevelSigma2(frame.mvLevelSigma2), mvInvLevelSigma2(frame.mvInvLevelSigma2) {
        for (int i = 0; i < FRAME_GRID_COLS; i++)
            for (int j = 0; j < FRAME_GRID_ROWS; j++)
                mGrid[i][j] = frame.mGrid[i][j];

        if (!frame.mTcw.empty())
            SetPose(frame.mTcw);
    }

    Frame::Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timeStamp, ORBextractor *extractorLeft,
                 ORBextractor *extractorRight, cv::Mat &K, const float &bf, const float &thDepth) :
            mpORBextractorLeft(extractorLeft), mpORBextractorRight(extractorRight), mTimeStamp(timeStamp),
            mK(K.clone()), mbf(bf), mThDepth(thDepth), mpReferenceKF(static_cast<KeyFrame*>(NULL)) {
        // Frame ID
        mnId = nNextId++;

        // Scale Level Info
        mnScaleLevels = mpORBextractorLeft->GetLevels();
        mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
        mfLogScaleFactor = log(mfScaleFactor);
        mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
        mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
        mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
        mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

        // ORB extraction
        thread threadLeft(&Frame::ExtractORB, this, 0, imLeft);
        thread threadRight(&Frame::ExtractORB, this, 1, imRight);
        threadLeft.join();
        threadRight.join();

        N = mvKeysLeft.size();
        if (mvKeysLeft.empty())   // No keypoints extracted from the left image
            return;

        mvbOutlier.resize(N, false);
        mvpMapPoints.resize(N, static_cast<MapPoint *>(NULL));
        mvpPlanes.resize(N, static_cast<Plane *>(NULL));

        // Compute matches between the left and right image
        ComputeStereoMatches();

        // This is done only for the first Frame
        if (mbInitialComputations) {
            mnMinX = 0;
            mnMinY = 0;
            mnMaxX = imLeft.cols;
            mnMaxY = imLeft.rows;

            mfGridElementWidthInv = static_cast<float>(FRAME_GRID_COLS) / (mnMaxX - mnMinX);
            mfGridElementHeightInv = static_cast<float>(FRAME_GRID_ROWS) / (mnMaxY - mnMinY);

            fx = K.at<float>(0, 0);
            fy = K.at<float>(1, 1);
            cx = K.at<float>(0, 2);
            cy = K.at<float>(1, 2);
            invfx = 1.0f / fx;
            invfy = 1.0f / fy;

            LOG(INFO) << fx << " " << fy << " " << cx << " " << cy << " " << mbf;

            mbInitialComputations = false;
        }

        mb = mbf / fx;

        AssignFeaturesToGrid();

        InitializeEpipolarPlane();
    }

    void Frame::ExtractORB(int flag, const cv::Mat &im) {
        if (flag == 0)
            (*mpORBextractorLeft)(im, cv::Mat(), mvKeysLeft, mDescriptorsLeft);
        else
            (*mpORBextractorRight)(im, cv::Mat(), mvKeysRight, mDescriptorsRight);
    }

    void Frame::SetPose(cv::Mat Tcw) {
        mTcw = Tcw.clone();
        UpdatePoseMatrices();

        // UpdatePlanes();
    }

    void Frame::UpdatePoseMatrices() {
        mRcw = mTcw.rowRange(0, 3).colRange(0, 3);
        mRwc = mRcw.t();
        mtcw = mTcw.rowRange(0, 3).col(3);
        mOw = -mRcw.t() * mtcw;
    }

    void Frame::UpdatePlanes() {
        for (int i = 0; i < N; ++i)
        {
            Plane* plane = mvpPlanes[i];
            if (plane)
            {
                plane->Transform(mRwc, mOw);
            }
        }
    }

    vector<size_t> Frame::GetFeaturesInArea(const float &x, const float &y, const float &r, const int minLevel,
                                            const int maxLevel) const {
        vector<size_t> vIndices;
        vIndices.reserve(N);

        const int nMinCellX = max(0, (int) floor((x - mnMinX - r) * mfGridElementWidthInv));
        if (nMinCellX >= FRAME_GRID_COLS)
            return vIndices;

        const int nMaxCellX = min((int) FRAME_GRID_COLS - 1, (int) ceil((x - mnMinX + r) * mfGridElementWidthInv));
        if (nMaxCellX < 0)
            return vIndices;

        const int nMinCellY = max(0, (int) floor((y - mnMinY - r) * mfGridElementHeightInv));
        if (nMinCellY >= FRAME_GRID_ROWS)
            return vIndices;

        const int nMaxCellY = min((int) FRAME_GRID_ROWS - 1, (int) ceil((y - mnMinY + r) * mfGridElementHeightInv));
        if (nMaxCellY < 0)
            return vIndices;

        const bool bCheckLevels = (minLevel > 0) || (maxLevel >= 0);

        for (int ix = nMinCellX; ix <= nMaxCellX; ix++) {
            for (int iy = nMinCellY; iy <= nMaxCellY; iy++) {
                const vector<size_t> vCell = mGrid[ix][iy];
                if (vCell.empty())
                    continue;

                for (size_t j = 0, jend = vCell.size(); j < jend; j++) {
                    const cv::KeyPoint &kpUn = mvKeysLeft[vCell[j]];
                    if (bCheckLevels) {
                        if (kpUn.octave < minLevel)
                            continue;
                        if (maxLevel >= 0)
                            if (kpUn.octave > maxLevel)
                                continue;
                    }

                    const float distx = kpUn.pt.x - x;
                    const float disty = kpUn.pt.y - y;

                    if (fabs(distx) < r && fabs(disty) < r)
                        vIndices.push_back(vCell[j]);
                }
            }
        }

        return vIndices;
    }

    void Frame::ComputeStereoMatches() {
        mvuRight = vector<float>(N, -1.0f);
        mvDepth = vector<float>(N, -1.0f);
        mnMatches = vector<int>(N, -1);

        const int thOrbDist = (ORBmatcher::TH_HIGH + ORBmatcher::TH_LOW) / 2;

        const int nRows = mpORBextractorLeft->mvImagePyramid[0].rows;

        //Assign keypoints to row table
        vector<vector<size_t> > vRowIndices(nRows, vector<size_t>());

        for (int i = 0; i < nRows; i++)
            vRowIndices[i].reserve(200);

        const int Nr = mvKeysRight.size();

        for (int iR = 0; iR < Nr; iR++) {
            const cv::KeyPoint &kp = mvKeysRight[iR];
            const float &kpY = kp.pt.y;
            const float r = 2.0f * mvScaleFactors[mvKeysRight[iR].octave];
            const int maxr = ceil(kpY + r);
            const int minr = floor(kpY - r);

            for (int yi = minr; yi <= maxr; yi++)
                vRowIndices[yi].push_back(iR);
        }

        // Set limits for search
        const float minZ = mb;
        const float minD = 0;
        const float maxD = mbf / minZ;

        // For each left keypoint search a match in the right image
        vector<pair<int, int> > vDistIdx;
        vDistIdx.reserve(N);

        for (int iL = 0; iL < N; iL++) {
            const cv::KeyPoint &kpL = mvKeysLeft[iL];
            const int &levelL = kpL.octave;
            const float &vL = kpL.pt.y;
            const float &uL = kpL.pt.x;

            const vector<size_t> &vCandidates = vRowIndices[vL];

            if (vCandidates.empty())
                continue;

            const float minU = uL - maxD;
            const float maxU = uL - minD;

            if (maxU < 0)
                continue;

            int bestDist = ORBmatcher::TH_HIGH;
            size_t bestIdxR = 0;

            const cv::Mat &dL = mDescriptorsLeft.row(iL);

            // Compare descriptor to right keypoints
            for (size_t iC = 0; iC < vCandidates.size(); iC++) {
                const size_t iR = vCandidates[iC];
                const cv::KeyPoint &kpR = mvKeysRight[iR];

                if (kpR.octave < levelL - 1 || kpR.octave > levelL + 1)
                    continue;

                const float &uR = kpR.pt.x;

                if (uR >= minU && uR <= maxU) {
                    const cv::Mat &dR = mDescriptorsRight.row(iR);
                    const int dist = ORBmatcher::DescriptorDistance(dL, dR);

                    if (dist < bestDist) {
                        bestDist = dist;
                        bestIdxR = iR;
                    }
                }
            }

            // Subpixel match by correlation
            if (bestDist < thOrbDist) {
                // coordinates in image pyramid at keypoint scale
                const float uR0 = mvKeysRight[bestIdxR].pt.x;
                const float scaleFactor = mvInvScaleFactors[kpL.octave];
                const float scaleduL = round(kpL.pt.x * scaleFactor);
                const float scaledvL = round(kpL.pt.y * scaleFactor);
                const float scaleduR0 = round(uR0 * scaleFactor);

                // sliding window search
                const int w = 5;
                cv::Mat IL = mpORBextractorLeft->mvImagePyramid[kpL.octave].rowRange(scaledvL - w,
                                                                                     scaledvL + w + 1).colRange(
                        scaleduL - w, scaleduL + w + 1);
                IL.convertTo(IL, CV_32F);
                IL = IL - IL.at<float>(w, w) * cv::Mat::ones(IL.rows, IL.cols, CV_32F);

                int bestDist = INT_MAX;
                int bestincR = 0;
                const int L = 5;
                vector<float> vDists;
                vDists.resize(2 * L + 1);

                const float iniu = scaleduR0 + L - w;
                const float endu = scaleduR0 + L + w + 1;
                if (iniu < 0 || endu >= mpORBextractorRight->mvImagePyramid[kpL.octave].cols)
                    continue;

                for (int incR = -L; incR <= +L; incR++) {
                    cv::Mat IR = mpORBextractorRight->mvImagePyramid[kpL.octave].rowRange(scaledvL - w,
                                                                                          scaledvL + w + 1).colRange(
                            scaleduR0 + incR - w, scaleduR0 + incR + w + 1);
                    IR.convertTo(IR, CV_32F);
                    IR = IR - IR.at<float>(w, w) * cv::Mat::ones(IR.rows, IR.cols, CV_32F);

                    float dist = cv::norm(IL, IR, cv::NORM_L1);
                    if (dist < bestDist) {
                        bestDist = dist;
                        bestincR = incR;
                    }

                    vDists[L + incR] = dist;
                }

                if (bestincR == -L || bestincR == L)
                    continue;

                // Sub-pixel match (Parabola fitting)
                const float dist1 = vDists[L + bestincR - 1];
                const float dist2 = vDists[L + bestincR];
                const float dist3 = vDists[L + bestincR + 1];

                const float deltaR = (dist1 - dist3) / (2.0f * (dist1 + dist3 - 2.0f * dist2));

                if (deltaR < -1 || deltaR > 1)
                    continue;

                // Re-scaled coordinate
                float bestuR = mvScaleFactors[kpL.octave] * ((float) scaleduR0 + (float) bestincR + deltaR);

                float disparity = (uL - bestuR);

                if (disparity >= minD && disparity < maxD) {
                    if (disparity <= 0) {
                        disparity = 0.01;
                        bestuR = uL - 0.01;
                    }
                    mvDepth[iL] = mbf / disparity;
                    mvuRight[iL] = bestuR;
                    mnMatches[iL] = bestIdxR;
                    vDistIdx.push_back(pair<int, int>(bestDist, iL));
                }
            }
        }

        sort(vDistIdx.begin(), vDistIdx.end());
        const float median = vDistIdx[vDistIdx.size() / 2].first;
        const float thDist = 1.5f * 1.4f * median;

        for (int i = vDistIdx.size() - 1; i >= 0; i--) {
            if (vDistIdx[i].first < thDist)
                break;
            else {
                mvuRight[vDistIdx[i].second] = -1;
                mvDepth[vDistIdx[i].second] = -1;
                mnMatches[vDistIdx[i].second] = -1;
            }
        }
    }

    cv::Mat Frame::UnprojectStereo(const int &i) {
        const float z = mvDepth[i];
        if (z > 0) {
            const float u = mvKeysLeft[i].pt.x;
            const float v = mvKeysLeft[i].pt.y;
            const float x = (u - cx) * z * invfx;
            const float y = (v - cy) * z * invfy;
            cv::Mat x3Dc = (cv::Mat_<float>(3, 1) << x, y, z);
            return mRwc * x3Dc + mOw;
        } else
            return cv::Mat();
    }


    void Frame::AssignFeaturesToGrid() {
        int nReserve = 0.5f * N / (FRAME_GRID_COLS * FRAME_GRID_ROWS);
        for (unsigned int i = 0; i < FRAME_GRID_COLS; i++)
            for (unsigned int j = 0; j < FRAME_GRID_ROWS; j++)
                mGrid[i][j].reserve(nReserve);

        for(int i=0;i<N;i++)
        {
            const cv::KeyPoint &kp = mvKeysLeft[i];

            int nGridPosX, nGridPosY;
            if(PosInGrid(kp,nGridPosX,nGridPosY))
                mGrid[nGridPosX][nGridPosY].push_back(i);
        }
    }

    bool Frame::PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY) {
        posX = round((kp.pt.x-mnMinX)*mfGridElementWidthInv);
        posY = round((kp.pt.y-mnMinY)*mfGridElementHeightInv);

        // Keypoint's coordinates are undistorted, which could cause to go out of the image
        if(posX<0 || posX>=FRAME_GRID_COLS || posY<0 || posY>=FRAME_GRID_ROWS)
            return false;

        return true;
    }

    void Frame::InitializeEpipolarPlane() {
        cv::Point3f pCl = cv::Point3f(0.0f, 0.0f, 0.0f);
        cv::Point3f pCr = cv::Point3f(mb, 0.0f, 0.0f);

        for (int i = 0; i < N; ++i)
        {
            const float z = mvDepth[i];
            if (z > 0) {
                const float u = mvKeysLeft[i].pt.x;
                const float v = mvKeysLeft[i].pt.y;
                const float x = (u - cx) * z * invfx;
                const float y = (v - cy) * z * invfy;
                cv::Point3f pX = cv::Point3f( x, y, z);

                Plane* newPlane = new Plane(pCl, pCr, pX);
                mvpPlanes[i] = newPlane;

            }
        }
    }

    float Frame::ComputeMeanReprojectionError() {
        float totalError = 0.0f;
        int nPoints = 0;

        for (int i = 0; i < N; ++i)
        {
            MapPoint* pMP = mvpMapPoints[i];
            if (!pMP || mvbOutlier[i]) continue;

            cv::Mat x3Dw = pMP->GetWorldPos();
            cv::Mat x3Dc = mRcw * x3Dw + mtcw;

            const float invz = 1.0f / x3Dc.at<float>(2);
            const float x = x3Dc.at<float>(0);
            const float y = x3Dc.at<float>(1);
            const float ru = fx * x * invz + cx;
            const float rv = fy * y * invz + cy;

            const cv::Point2f& p = mvKeysLeft[i].pt;
            const float disp = mbf * invz;
            const float ud = p.x - mvuRight[i];

            // Error is defined on the stereo frame
            totalError += sqrt((ru - p.x) * (ru - p.x) + (rv - p.y) * (rv - p.y)); // + (disp - ud) * (disp - ud));
            nPoints++;
        }

        return totalError / nPoints;
    }

    float Frame::ComputeMeanDistanceToPlane() {
        float totalError = 0.0;
        int nPlanes = 0;

        for (int i = 0; i < N; ++i)
        {
            MapPoint* pMP = mvpMapPoints[i];
            Plane* plane = mvpPlanes[i];

            if (!pMP || mvbOutlier[i]) continue;

            cv::Mat x3Dw = pMP->GetWorldPos();
            totalError += plane->ComputeDistance(x3Dw);
            nPlanes++;
        }

        return totalError / nPlanes;
    }

    bool Frame::IsInFrustum(MapPoint *pMP, float view) {
        pMP->mbTrackInView = false;

        // 3D in absolute coordinates
        cv::Mat P = pMP->GetWorldPos();

        // LOG(INFO) << "mType: " << pMP->mType << " " << P << " " << mRcw << " " << mtcw;

        // 3D in camera coordinates
        const cv::Mat Pc = mRcw*P+mtcw;
        const float &PcX = Pc.at<float>(0);
        const float &PcY= Pc.at<float>(1);
        const float &PcZ = Pc.at<float>(2);

        // Check positive depth
        if(PcZ<0.0f)
            return false;

        // Project in image and check it is not outside
        const float invz = 1.0f/PcZ;
        const float u=fx*PcX*invz+cx;
        const float v=fy*PcY*invz+cy;

        if(u<mnMinX || u>mnMaxX)
            return false;
        if(v<mnMinY || v>mnMaxY)
            return false;

        // Check distance is in the scale invariance region of the MapPoint
        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        const cv::Mat PO = P-mOw;
        const float dist = cv::norm(PO);

        if(dist<minDistance || dist>maxDistance)
            return false;

        // Check viewing angle
        cv::Mat Pn = pMP->GetNormal();

        const float viewCos = PO.dot(Pn)/dist;

//        if(viewCos<viewingCosLimit)
//            return false;

        // Predict scale in the image
        const int nPredictedLevel = pMP->PredictScale(dist,this);

        // Data used by the tracking
        pMP->mbTrackInView = true;
        pMP->mTrackProjX = u;
        pMP->mTrackProjXR = u - mbf*invz;
        pMP->mTrackProjY = v;
        pMP->mnTrackScaleLevel= nPredictedLevel;
        pMP->mTrackViewCos = viewCos;

        return true;
    }

}

