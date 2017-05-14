//
// Created by feixue on 17-5-4.
//

#ifndef EPIPOLARPLANEMATCH_MAP_H
#define EPIPOLARPLANEMATCH_MAP_H

#include <KeyFrame.h>
#include <MapPoint.h>
#include <Plane.h>

namespace EpipolarPlaneMatch
{
    class KeyFrame;
    class Map
    {
    public:
        Map();
        ~Map();

    public:
        void AddMapPoint(MapPoint* pMP);
        void AddPlane(Plane* pPlane);
        void AddKeyFrame(KeyFrame* pKF);

        void EraseMapPoint(MapPoint* pMP);
        void ErasePlane(Plane* pPlane);
        void EraseKeyFrame(KeyFrame* pKF);

        std::vector<KeyFrame*> GetAllKeyFrames();
        std::vector<MapPoint*> GetAllMapPoints();
        std::vector<Plane*> GetAllPlanes();

        float ComputeMeanReprojectionError();

    protected:
        float ReprojectionError(const cv::Mat& X3D, const cv::Point2f& pt, const float& ru, const float& fx, const float& fy,
                                const float& cx, const float& cy, const float& bf);

    protected:
        std::set<MapPoint*> mspMapPoints;
        std::set<Plane*> mspPlanes;
        std::set<KeyFrame*> mspKeyFrames;
    };
}
#endif //EPIPOLARPLANEMATCH_MAP_H
