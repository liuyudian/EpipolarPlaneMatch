//
// Created by feixue on 17-4-29.
//

#ifndef EPIPOLARPLANEMATCH_VISUALIZER_H
#define EPIPOLARPLANEMATCH_VISUALIZER_H

#include <Plane.h>
#include <opencv2/viz/viz3d.hpp>

#include <glog/logging.h>
#include <mutex>
namespace EpipolarPlaneMatch
{
//    class WTriangulation:public cv::viz::Widget3D
//    {
//    public:
//        WTriangulation(const cv::Point3d &pt1, const cv::Point3d &pt2, const cv::Point3d& pt3, const cv::viz::Color &color = cv::viz::Color::white());
//    };

    class Visualizer
    {
    public:
        Visualizer();
        ~Visualizer();

    public:
        void Run();

        void DrawCameraPoses(std::vector<cv::Mat>& cameras);

        void DrawEpipolarPlanes(std::vector<Plane>& planes);

        void AddCamera(cv::Mat& camera);
        void AddPlane(Plane* mpPlane);
        

    protected:
        // Visualizer components
        cv::viz::Viz3d mWindow;

        int nWaitTime;

        // Data
        std::vector<cv::Mat> mvCameras;
        std::vector<Plane*> mvpPlanes;

        std::mutex mMutexCameraAndPlane;

    };
}

#endif //EPIPOLARPLANEMATCH_VISUALIZER_H
