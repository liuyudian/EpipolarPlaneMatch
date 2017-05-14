//
// Created by feixue on 17-5-4.
//

#include <Map.h>

namespace EpipolarPlaneMatch
{
    Map::Map() {}
    Map::~Map() {}

    void Map::AddMapPoint(MapPoint *pMP) {
        if (!pMP) return;

        mspMapPoints.insert(pMP);
    }

    void Map::AddPlane(Plane *pPlane) {
        if (!pPlane) return;

        mspPlanes.insert(pPlane);
    }

    void Map::AddKeyFrame(KeyFrame *pKF) {
        if (!pKF) return;

        mspKeyFrames.insert(pKF);
    }

    void Map::EraseKeyFrame(KeyFrame *pKF) {
        if (!pKF) return;

        mspKeyFrames.erase(pKF);
    }

    void Map::EraseMapPoint(MapPoint *pMP) {
        if (!pMP) return;

        mspMapPoints.erase(pMP);
    }

    void Map::ErasePlane(Plane *pPlane) {
        if (!pPlane) return;

        mspPlanes.erase(pPlane);
    }

    std::vector<MapPoint*> Map::GetAllMapPoints() {

        return std::vector<MapPoint*>(mspMapPoints.begin(), mspMapPoints.end());
    }

    std::vector<Plane*> Map::GetAllPlanes() {
        return std::vector<Plane*>(mspPlanes.begin(), mspPlanes.end());
    }

    std::vector<KeyFrame*> Map::GetAllKeyFrames() {

        return std::vector<KeyFrame*>(mspKeyFrames.begin(), mspKeyFrames.end());
    }

    float Map::ComputeMeanReprojectionError() {

        std::vector<MapPoint*> vPoints = std::vector<MapPoint*>(mspMapPoints.begin(), mspMapPoints.end());

        int nPoints = 0;
        double totalError = 0;
        for (int i = 0, iend = vPoints.size(); i < iend; ++i)
        {
            MapPoint* pMP = vPoints[i];
            if (pMP->IsBad()) continue;

            std::map<KeyFrame*, size_t> obs = pMP->GetAllObservations();
            for (std::map<KeyFrame*, size_t>::iterator mit = obs.begin(), mend = obs.end(); mit != mend; mit++)
            {
                KeyFrame* pKF = mit->first;
                size_t idx = mit->second;

                const float& fx = pKF->fx;
                const float& fy = pKF->fy;
                const float& cx = pKF->cx;
                const float& cy = pKF->cy;
                const float& bf = pKF->mbf;

                const cv::Point2f& pt = pKF->mvKeys[idx].pt;
                const float ru = pKF->mvuRight[idx];

                float error = ReprojectionError(pMP->GetWorldPos(), pt, ru, fx, fy, cx, cy, bf);
                totalError += error;

                nPoints++;
            }
        }
        if (nPoints == 0) return 0;

        return totalError / nPoints;
    }

    float Map::ReprojectionError(const cv::Mat &X3D, const cv::Point2f &pt, const float &ru, const float &fx,
                                 const float &fy, const float &cx, const float &cy, const float &bf) {

        const float& invz = 1.0 / X3D.at<float>(2);
        const float& pre_u = X3D.at<float>(0) * fx * invz + cx;
        const float& pre_v = X3D.at<float>(1) * fy * invz + cy;

        const float& d = bf* invz;

        const float& du = pre_u - pt.x;
        const float& dv = pre_v - pt.y;
        const float& dd = d - (pt.x - ru);

        return sqrt(du * du + dv * dv + dd * dd);
    }
}