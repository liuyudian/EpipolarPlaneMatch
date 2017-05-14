//
// Created by feixue on 17-4-29.
//

#include <Plane.h>
#include <glog/logging.h>

namespace EpipolarPlaneMatch
{
    unsigned long Plane::mnNext = 0;

    Plane::Plane() {}

    Plane::Plane(const cv::Mat &normal, const float &d)
    {
        mnId = mnNext++;

        mNormal = normal.clone();
        mD = d;
    }

    Plane::Plane(const cv::Point3f &p1, const cv::Point3f &p2, const cv::Point3f &p3)
    {
        // Homogeneous coordinate
        cv::Mat X1 = (cv::Mat_<float>(3, 1) << p1.x, p1.y, p1.z);
        cv::Mat X2 = (cv::Mat_<float>(3, 1) << p2.x, p2.y, p2.z);
        cv::Mat X3 = (cv::Mat_<float>(3, 1) << p3.x, p3.y, p3.z);

        cv::Mat X12 = X2 - X1;
        cv::Mat X13 = X3 - X1;
//
//        LOG(INFO) << X12;
//        LOG(INFO) << X13;

        cv::Mat normal = X12.cross(X13);

        mNormal = normal / cv::norm(normal);         // the unit normal vector
        mD = cv::norm(X1.t() * mNormal);  // the distance to the origin

        mRawNormal = mNormal.clone();
        mRawD = mD;

//        LOG(INFO) << "mNormal: " << mNormal;
//        LOG(INFO) << "mD: " << mD;
    }

    Plane::~Plane()
    {
    }

    void Plane::SetNormalAndDistance(const cv::Mat &normal, const float &d)
    {
        assert(d >= 0);
        assert(normal.type() == CV_32F);

        mNormal = normal.clone();
        mD = d;
    }

    void Plane::Transform(const cv::Mat &R, const cv::Mat &t) {
        // Update normal vector and distance to the origin
        // n' = R * n
        // d' = -t.t()*n' + d
        if (R.empty() || t.empty())
            return;

        mNormal = R * mRawNormal;
        mD = fabsf(-mNormal.dot(t) + mRawD);

        // LOG(INFO) << "mD: " << mD << " , raw D: " << mRawD;
    }

    void Plane::Transform(const cv::Mat &T) {
        if (T.empty())
            return;

        // Extract R and t
        const cv::Mat R = T.rowRange(0, 3).colRange(0, 3).clone();
        const cv::Mat t = T.rowRange(0, 3).col(3);

        Transform(R, t);
    }

    cv::Mat Plane::GetNormal() {
        return mNormal.clone();
    }

    cv::Mat Plane::GetRawNormal() {
        return mRawNormal;
    }

    float Plane::GetDistance() {
        return mD;
    }

    float Plane::GetRawDistance() {
        return mRawD;
    }

    float Plane::ComputeDistance(const cv::Mat &x3Dw) {
        if (x3Dw.empty()) return -1;

        // LOG(INFO) << mNormal.dot(x3Dw);
        float dist = fabsf(mD - mNormal.dot(x3Dw));

        return dist;
    }


}