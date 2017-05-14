//
// Created by feixue on 17-4-30.
//

#include <KeyFrame.h>

using namespace std;

namespace EpipolarPlaneMatch
{
    long unsigned int KeyFrame::nNextId = 0;

    KeyFrame::KeyFrame(Frame &frame): mnFrameId(frame.mnId), N(frame.N), mTimeStamp(frame.mTimeStamp), mnGridCols(FRAME_GRID_COLS), mnGridRows(FRAME_GRID_ROWS),
                                      mfGridElementWidthInv(frame.mfGridElementWidthInv), mfGridElementHeightInv(frame.mfGridElementHeightInv), fx(frame.fx), fy(frame.fy), cx(frame.cx), cy(frame.cy),
                                      invfx(frame.invfx), invfy(frame.invfy), mbf(frame.mbf), mb(frame.mb), mHalfBaseline(mb / 2), mThDepth(frame.mThDepth), mvKeys(frame.mvKeysLeft), mvuRight(frame.mvuRight),
                                      mvDepth(frame.mvDepth), mDescriptors(frame.mDescriptorsLeft.clone()), mnScaleLevels(frame.mnScaleLevels), mfScaleFactor(frame.mfScaleFactor),
                                      mfLogScaleFactor(frame.mfLogScaleFactor), mvScaleFactors(frame.mvScaleFactors), mvLevelSigma2(frame.mvLevelSigma2),
                                      mvInvLevelSigma2(frame.mvInvLevelSigma2), mnMinX(frame.mnMinX), mnMinY(frame.mnMinY), mnMaxX(frame.mnMaxX),
                                      mnMaxY(frame.mnMaxY), mK(frame.mK.clone()), mnBALocalKF(0), mnTrackReferenceForFrame(frame.mnId)

    {
        mnId=nNextId++;

        mvpMapPoints.resize(N, static_cast<MapPoint*>(NULL));
        mvpPlanes.resize(N, static_cast<Plane*>(NULL));

        mGrid.resize(mnGridCols);
        for(int i=0; i<mnGridCols;i++)
        {
            mGrid[i].resize(mnGridRows);
            for(int j=0; j<mnGridRows; j++)
                mGrid[i][j] = frame.mGrid[i][j];
        }

        SetPose(frame.mTcw);
    }

    void KeyFrame::SetPose(const cv::Mat &Tcw_)
    {
        Tcw_.copyTo(Tcw);
        cv::Mat Rcw = Tcw.rowRange(0,3).colRange(0,3);
        cv::Mat tcw = Tcw.rowRange(0,3).col(3);
        cv::Mat Rwc = Rcw.t();
        Ow = -Rwc*tcw;

        Twc = cv::Mat::eye(4,4,Tcw.type());
        Rwc.copyTo(Twc.rowRange(0,3).colRange(0,3));
        Ow.copyTo(Twc.rowRange(0,3).col(3));
        cv::Mat center = (cv::Mat_<float>(4,1) << mHalfBaseline, 0 , 0, 1);
        Cw = Twc*center;
    }

    cv::Mat KeyFrame::GetPose() {
        return Tcw.clone();
    }

    cv::Mat KeyFrame::GetPoseInverse() {
        return Twc.clone();
    }

    cv::Mat KeyFrame::GetCameraCenter() {
        return Ow.clone();
    }


    cv::Mat KeyFrame::GetRotation()
    {
        return Tcw.rowRange(0,3).colRange(0,3).clone();
    }

    cv::Mat KeyFrame::GetTranslation()
    {
        return Tcw.rowRange(0,3).col(3).clone();
    }

    void KeyFrame::AddMapPoint(MapPoint *pMP, const size_t &idx) {
        mvpMapPoints[idx] = pMP;
    }

    void KeyFrame::EraseMapPointMatch(MapPoint *pMP) {
        int idx = pMP->GetIndexInKeyFrame(this);
        if (idx >= 0)
            mvpMapPoints[idx] = static_cast<MapPoint*>(NULL);
    }

    void KeyFrame::EraseMapPointMatch(const int &idx) {
        if (idx < 0) return;

        mvpMapPoints[idx] = static_cast<MapPoint*>(NULL);
    }

    void KeyFrame::AddPlane(Plane *pPlane, const size_t &idx) {
        mvpPlanes[idx] = pPlane;
    }

    std::vector<MapPoint*> KeyFrame::GetMapPointMatches() {
        return mvpMapPoints;
    }

    std::vector<MapPoint*> KeyFrame::GetValidMapPoints() {
        std::vector<MapPoint*> vMapPoints;

        for (int i = 0; i < N; ++i)
        {
            MapPoint* pMP = mvpMapPoints[i];
            if (pMP)                           // More conditions
                vMapPoints.push_back(pMP);
        }

        return vMapPoints;
    }

    MapPoint* KeyFrame::GetMapPoint(const size_t &idx) {
        return mvpMapPoints[idx];
    }

    std::vector<Plane*> KeyFrame::GetPlaneMatches() {
        return mvpPlanes;
    }

    Plane* KeyFrame::GetPlane(const size_t &idx) {
        return mvpPlanes[idx];
    }

    std::vector<Plane*> KeyFrame::GetValidPlanes() {
        std::vector<Plane*> vPlanes;

        for (int i = 0; i < N; ++i)
        {
            Plane* plane = mvpPlanes[i];
            if (plane)
                vPlanes.push_back(plane);
        }

        return vPlanes;
    }

    void KeyFrame::UpdateConnections() {
        std::map<KeyFrame*, int> KFcounter;

        std::vector<MapPoint*> vpMPs;

        for (std::vector<MapPoint*>::iterator vit = vpMPs.begin(), vend = vpMPs.end(); vit != vend; vit++)
        {
            MapPoint* pMP = *vit;
            if (!pMP) continue;

            std::map<KeyFrame*, size_t > observations = pMP->GetAllObservations();

            for (std::map<KeyFrame*, size_t>::iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
            {
                if (mit->first->mnId == mnId)
                    continue;
                KFcounter[mit->first]++;
            }
        }

        if (KFcounter.empty())
            return;

        int nmax = 0;
        KeyFrame* pKFmax = NULL;
        int th = 15;

        std::vector<std::pair<int, KeyFrame*> > vPairs;
        vPairs.reserve(KFcounter.size());

        for (std::map<KeyFrame*, int>::iterator mit = KFcounter.begin(), mend = KFcounter.end(); mit != mend; mit++)
        {
            if (mit->second > nmax)
            {
                nmax = mit->second;
                pKFmax = mit->first;
            }
            if (mit->second > th)
            {
                vPairs.push_back(std::make_pair(mit->second, mit->first));
                (mit->first)->AddConnection(this, mit->second);
            }
        }

        if (vPairs.empty())
        {
            vPairs.push_back(std::make_pair(nmax, pKFmax));
            pKFmax->AddConnection(this, nmax);
        }

        std::sort(vPairs.begin(), vPairs.end());
        std::list<KeyFrame*> lKFs;
        std::list<int> lWs;
        for (size_t i = 0; i < vPairs.size(); ++i)
        {
            lKFs.push_front(vPairs[i].second);
            lWs.push_front(vPairs[i].first);
        }

        mConnectedKeyFrameWeights = KFcounter;
        mvpOrderedConnectedKeyFrames = std::vector<KeyFrame*>(lKFs.begin(), lKFs.end());
        mvOrderedWeights = std::vector<int>(lWs.begin(), lWs.end());
    }

    void KeyFrame::AddConnection(KeyFrame *pKF, const int weight) {
        if (!mConnectedKeyFrameWeights.count(pKF))
            mConnectedKeyFrameWeights[pKF] = weight;
        else if (mConnectedKeyFrameWeights[pKF] != weight)
            mConnectedKeyFrameWeights[pKF] = weight;
        else
            return;

        UpdateBestCovisibility();

    }

    void KeyFrame::UpdateBestCovisibility() {

        std::vector<std::pair<int, KeyFrame*> > vPairs;
        vPairs.reserve(mConnectedKeyFrameWeights.size());
        for (std::map<KeyFrame*, int>::iterator mit = mConnectedKeyFrameWeights.begin(), mend = mConnectedKeyFrameWeights.end(); mit != mend; mit++)
            vPairs.push_back(std::make_pair(mit->second, mit->first));

        std::sort(vPairs.begin(), vPairs.end());

        std::list<KeyFrame*> lKFs;
        std::list<int> lWs;
        for (size_t i = 0; i < vPairs.size(); ++i)
        {
            lKFs.push_front(vPairs[i].second);
            lWs.push_front(vPairs[i].first);
        }

        mvpOrderedConnectedKeyFrames = std::vector<KeyFrame*>(lKFs.begin(), lKFs.end());
        mvOrderedWeights = std::vector<int>(lWs.begin(), lWs.end());
    }

    std::vector<KeyFrame*> KeyFrame::GetVectorCovisibleKeyFrames() {
        return mvpOrderedConnectedKeyFrames;
    }

    std::vector<KeyFrame*> KeyFrame::GetBestCovisibleKeyFrames(const int &n) {
        if (mvpOrderedConnectedKeyFrames.size() < n)
            return mvpOrderedConnectedKeyFrames;

        return std::vector<KeyFrame*>(mvpOrderedConnectedKeyFrames.begin(), mvpOrderedConnectedKeyFrames.begin() + n);
    }

    bool KeyFrame::IsBad() const {
        return bBad;
    }


}
