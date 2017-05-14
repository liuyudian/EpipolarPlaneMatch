//
// Created by feixue on 17-4-30.
//

#include <MapPoint.h>
#include <include/ORBmatcher.h>
#include <glog/logging.h>

namespace EpipolarPlaneMatch
{
    unsigned long MapPoint::nNextId = 0;
    MapPoint::MapPoint() {}

    MapPoint::MapPoint(const cv::Mat &Pos, Frame *pFrame, const int &idxF)
            :mnFirstFrame(idxF), mnBALocalKF(0), bBad(false), mnVisible(0), mnFound(2), nObs(0)
    {
        Pos.copyTo(mWorldPos);
        cv::Mat Ow = pFrame->GetCameraCenter();
        // LOG(INFO) << "Ow" << Ow;

        // LOG(INFO) << mWorldPos.size() << " " << Ow.size();

        mNormalVector = mWorldPos - Ow;
        mNormalVector = mNormalVector / cv::norm(mNormalVector);

        cv::Mat PC = Pos - Ow;
        const float dist = cv::norm(PC);
        const int level = pFrame->mvKeysLeft[idxF].octave;
        const float levelScaleFactor = pFrame->mvScaleFactors[level];
        const int nLevels = pFrame->mnScaleLevels;

        mfMaxDistance = dist * levelScaleFactor;
        mfMinDistance = mfMaxDistance / pFrame->mvScaleFactors[nLevels - 1];

        pFrame->mDescriptorsLeft.row(idxF).copyTo(mDescriptor);

        mnId = nNextId++;
    }

    void MapPoint::SetWorldPos(cv::Mat Pos) {
        Pos.copyTo(mWorldPos);
    }

    cv::Mat MapPoint::GetWorldPos() {
        return mWorldPos.clone();
    }

    cv::Mat MapPoint::GetNormal() {
        return mNormalVector.clone();
    }

    cv::Mat MapPoint::GetDescriptor() {
        return mDescriptor.clone();
    }

    void MapPoint::AddObservation(KeyFrame *pKF, const size_t&idx) {
        if (!pKF || idx < 0) return;

        if (mObservations.count(pKF))  // Already exist
            return;

        mObservations[pKF] = idx;
        nObs += 1;
    }

    void MapPoint::EraseObservation(KeyFrame *pKF) {
        if (!mObservations.count(pKF)) return;

        mObservations.erase(pKF);
    }

    std::map<KeyFrame* ,size_t> MapPoint::GetAllObservations() {
        return mObservations;
    }


    void MapPoint::ComputeDistinctiveDescriptors() {
        // Retrieve all observed descriptors
        std::vector<cv::Mat> vDescriptors;

        std::map<KeyFrame *, size_t> observations;

        {
//            unique_lock<mutex> lock1(mMutexFeatures);
//            if (mbBad)
//                return;
            observations = mObservations;
        }

        if (observations.empty())
            return;

        vDescriptors.reserve(observations.size());

        for (std::map<KeyFrame *, size_t>::iterator mit = observations.begin(), mend = observations.end();
             mit != mend; mit++) {
            KeyFrame *pKF = mit->first;

            // if (!pKF->isBad())
                vDescriptors.push_back(pKF->mDescriptors.row(mit->second));
        }

        if (vDescriptors.empty())
            return;

        // Compute distances between them
        const size_t N = vDescriptors.size();

        float Distances[N][N];
        for (size_t i = 0; i < N; i++) {
            Distances[i][i] = 0;
            for (size_t j = i + 1; j < N; j++) {
                int distij = ORBmatcher::DescriptorDistance(vDescriptors[i], vDescriptors[j]);
                Distances[i][j] = distij;
                Distances[j][i] = distij;
            }
        }

        // Take the descriptor with least median distance to the rest
        int BestMedian = INT_MAX;
        int BestIdx = 0;
        for (size_t i = 0; i < N; i++) {
            std::vector<int> vDists(Distances[i], Distances[i] + N);
            std::sort(vDists.begin(), vDists.end());
            int median = vDists[0.5 * (N - 1)];

            if (median < BestMedian) {
                BestMedian = median;
                BestIdx = i;
            }
        }

        {
            // unique_lock<mutex> lock(mMutexFeatures);
            mDescriptor = vDescriptors[BestIdx].clone();
        }
    }

    void MapPoint::UpdateNormalAndDepth() {
        std::map<KeyFrame *, size_t> observations;
        KeyFrame *pRefKF;
        cv::Mat Pos;
        {
//            unique_lock<mutex> lock1(mMutexFeatures);
//            unique_lock<mutex> lock2(mMutexPos);
//            if (mbBad)
//                return;
            observations = mObservations;
            pRefKF = mpRefKF;
            Pos = mWorldPos.clone();
        }

        if (observations.empty())
            return;

        cv::Mat normal = cv::Mat::zeros(3, 1, CV_32F);
        int n = 0;
        for (std::map<KeyFrame *, size_t>::iterator mit = observations.begin(), mend = observations.end();
             mit != mend; mit++) {
            KeyFrame *pKF = mit->first;
            cv::Mat Owi = pKF->GetCameraCenter();
            cv::Mat normali = mWorldPos - Owi;
            normal = normal + normali / cv::norm(normali);
            n++;
        }

        cv::Mat PC = Pos - pRefKF->GetCameraCenter();
        const float dist = cv::norm(PC);
        const int level = pRefKF->mvKeys[observations[pRefKF]].octave;
        const float levelScaleFactor = pRefKF->mvScaleFactors[level];
        const int nLevels = pRefKF->mnScaleLevels;

        {
            // unique_lock<mutex> lock3(mMutexPos);
            mfMaxDistance = dist * levelScaleFactor;
            mfMinDistance = mfMaxDistance / pRefKF->mvScaleFactors[nLevels - 1];
            mNormalVector = normal / n;
        }
    }

    void MapPoint::SetReferenceKeyFrame(KeyFrame *pKF) {
        mpRefKF = pKF;
    }

    int MapPoint::GetIndexInKeyFrame(KeyFrame *pKF) {
        if (mObservations.count(pKF) <= 0) return -1;
        return mObservations[pKF];
    }

    void MapPoint::SetBad() {
        bBad = true;
    }

    bool MapPoint::IsBad() const {
        return bBad;
    }

    void MapPoint::IncreaseVisible(const int &n) {
        mnVisible += n;
    }

    void MapPoint::IncreaseFound(const int &n) {
        mnFound += n;
    }

    int MapPoint::Observations() {
        return nObs;
    }

    float MapPoint::GetMinDistanceInvariance() {
        // unique_lock<mutex> lock(mMutexPos);
        return 0.8f * mfMinDistance;
    }

    float MapPoint::GetMaxDistanceInvariance() {
        // unique_lock<mutex> lock(mMutexPos);
        return 1.2f * mfMaxDistance;
    }

    int MapPoint::PredictScale(const float &currentDist, Frame *pF) {
        float ratio;
        {
            // unique_lock<mutex> lock(mMutexPos);
            ratio = mfMaxDistance / currentDist;
        }

        int nScale = ceil(log(ratio) / pF->mfLogScaleFactor);
        if (nScale < 0)
            nScale = 0;
        else if (nScale >= pF->mnScaleLevels)
            nScale = pF->mnScaleLevels - 1;

        return nScale;
    }

    int MapPoint::PredictScale(const float &currentDist, KeyFrame *pKF) {
        float ratio;
        {
            // unique_lock<mutex> lock(mMutexPos);
            ratio = mfMaxDistance / currentDist;
        }

        int nScale = ceil(log(ratio) / pKF->mfLogScaleFactor);
        if (nScale < 0)
            nScale = 0;
        else if (nScale >= pKF->mnScaleLevels)
            nScale = pKF->mnScaleLevels - 1;

        return nScale;
    }





}

