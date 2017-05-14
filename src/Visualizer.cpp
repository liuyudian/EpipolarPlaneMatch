//
// Created by feixue on 17-4-29.
//

#include <Visualizer.h>

using namespace cv;
using namespace cv::viz;

namespace EpipolarPlaneMatch
{
    Visualizer::Visualizer()
    {
        cv::viz::Viz3d window1("PlaneWindow");
        cv::viz::Viz3d window2("CameraWindow");
    }

    Visualizer::~Visualizer() {

    }

    void Visualizer::Run() {
//        mWindow = cv::viz::getWindowByName("CameraWidow");
//        mWindow.showWidget("Coordinate Widget", cv::viz::WCoordinateSystem());

//        Plane plane(cv::Point3f(0, 0, 5), cv::Point3f(5, 0, 5), cv::Point3f(0, 0, 5))
//        cv::viz::WPlane wPlane(cv::Point3f(0, 0, 0), plane.GetNormal(), )

        Viz3d viz("show_simple_widgets");
        viz.setBackgroundMeshLab();

        cv::viz::WLine line1(cv::Point3f(0.0f, 0.0f, 0.0f), cv::Point3f(0.0f, 0.0f, 5.0f));
        cv::viz::WLine line2(cv::Point3f(0.0f, 0.0f, 0.0f), cv::Point3f(2.0f, 0.0f, 0.0f));
        cv::viz::WLine line3(cv::Point3f(2.0f, 0.0f, 0.0f), cv::Point3f(0.0f, 0.0f, 5.0f));
        // axis.setRenderingProperty(cv::viz::LINE_WIDTH, 4.0);
        viz.showWidget("line1", line1);
        viz.showWidget("line2", line2);
        viz.showWidget("line3", line3);
        viz.showWidget("Coordinate Widget", cv::viz::WCoordinateSystem());
        viz.showWidget("plane1", WPlane(cv::Size2d(0.25, 0.75)));
        viz.showWidget("plane2", WPlane(cv::Vec3d(0.5, -0.5, -0.5), cv::Vec3d(0.0, 1.0, 1.0), cv::Vec3d(1.0, 1.0, 0.0)));

        int i = 0;
        while (!viz.wasStopped())
        {
            viz.spinOnce(1, true);
        }
        viz.spin();
    }

    void Visualizer::DrawCameraPoses(std::vector<cv::Mat> &cameras) {

    }

    void Visualizer::DrawEpipolarPlanes(std::vector<Plane> &planes) {

    }

    void Visualizer::AddCamera(cv::Mat &camera) {

        std::unique_lock<std::mutex> lock(mMutexCameraAndPlane);
        mvCameras.push_back(camera);
    }

    void Visualizer::AddPlane(Plane *mpPlane) {
        std::unique_lock<std::mutex> lock(mMutexCameraAndPlane);
        mvpPlanes.push_back(mpPlane);
    }
}
