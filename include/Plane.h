//
// Created by feixue on 17-4-29.
//

#ifndef EPIPOLARPLANEMATCH_PLANE_H
#define EPIPOLARPLANEMATCH_PLANE_H

#include <opencv2/opencv.hpp>

namespace EpipolarPlaneMatch {
    class Plane {
    public:
        // The constructor function
        Plane();

        // Plane from 3 points
        Plane(const cv::Point3f &p1, const cv::Point3f &p2, const cv::Point3f &p3);

        // Plane from normal vector and distance to the origin
        Plane(const cv::Mat &normal, const float &d);

        ~Plane();

    public:
        void SetNormalAndDistance(const cv::Mat &normal, const float &d);

        void Transform(const cv::Mat& R, const cv::Mat& t);

        void Transform(const cv::Mat& T);

        cv::Mat GetNormal();

        cv::Mat GetRawNormal();

        float GetDistance();

        float GetRawDistance();

        /**
         * Compute distance from point to the plane
         * @param x3Dw a 3x1 mat
         * @return distance
         */
        float ComputeDistance(const cv::Mat& x3Dw);

    protected:
        // TODO

    public:
        unsigned long mnId;           // Plane id
        static unsigned long mnNext;

    protected:
        cv::Mat mNormal;   // 3x1, the normal vector of the plane
        float mD;          // The distance to the origin

        cv::Mat mRawNormal;  // 3x1, the raw normal vector of the plane
        float mRawD;         // The raw distance to the origin

        // Three 3D points that defines the plane
        cv::Mat mX;
        cv::Mat mOl;
        cv::Mat mOr;
    };
}

#endif //EPIPOLARPLANEMATCH_PLANE_H
