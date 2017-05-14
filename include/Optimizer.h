//
// Created by feixue on 17-5-3.
//

#ifndef EPIPOLARPLANEMATCH_OPTIMIZER_H
#define EPIPOLARPLANEMATCH_OPTIMIZER_H

#include <Frame.h>
#include <KeyFrame.h>
#include <MapPoint.h>

#include <opencv2/opencv.hpp>

#include "Thirdparty/g2o/g2o/types/types_seven_dof_expmap.h"

#include <ceres/ceres.h>
#include <ceres/solver.h>
#include <ceres/rotation.h>
#include <ceres/autodiff_cost_function.h>
#include <ceres/cost_function.h>

namespace EpipolarPlaneMatch
{
    typedef Eigen::Matrix<double, 3, 3> Mat3;
    typedef Eigen::Matrix<double, 6, 1> Vec6;
    typedef Eigen::Vector3d Vec3;
    typedef Eigen::Vector4d Vec4;

    struct ReprojectionErrorWithFixedPoint
    {
        ReprojectionErrorWithFixedPoint(const double _fx, const double _fy, const double _cx, const double _cy, const double _bf,
                                        const double _x, const double _y, const double _z,
                                        const double _obsX, const double _obsY, const double _obsRX)
                : fx(_fx), fy(_fy), cx(_cx), cy(_cy), bf(_bf),
                  x(_x), y(_y), z(_z),
                  obsX(_obsX), obsY(_obsY), obsRX(_obsRX){}

        template <typename T>
        bool operator() (const T* const camera, T* residuals) const {
            T p[3];
            T point[3] = {T(x), T(y), T(z)};
            ceres::AngleAxisRotatePoint(camera, point, p);

            // camera[3,4,5] are the translation
            p[0] += camera[3];
            p[1] += camera[4];
            p[2] += camera[5];

            const T& invz = 1.0 / p[2];
            const T& u = fx * p[0] * invz + cx;
            const T& v = fy * p[1] * invz + cy;
            const T& dis = bf / p[2];

            residuals[0] = sqrt((u - T(obsX)) * (u - T(obsX))+ (v - T(obsY)) * (v - T(obsY))); // + (dis - T(obsX - obsRX)) * (dis - T(obsX - obsRX)));
            // residuals[1] = sqrt((v - T(obsY)) * (v - T(obsY)));
            // residuals[2] = sqrt((dis - T(obsX - obsRX)) * (dis - T(obsX - obsRX)));


            return true;
        }

        // Factory to hide the construction of the CostFunction object from the client code.
        static ceres::CostFunction* Create(const double _fx, const double _fy, const double _cx, const double _cy, const double _bf,
                                           const double _x, const double _y, const double _z,
                                           const double _obsX, const double _obsY, const double _obsRX)
        {
            return (new ceres::AutoDiffCostFunction<ReprojectionErrorWithFixedPoint, 1, 6>(
                    new ReprojectionErrorWithFixedPoint(_fx, _fy, _cx, _cy, _bf, _x, _y, _z, _obsX, _obsY, _obsRX)));
        }

        double fx, fy, cx, cy, bf;
        double x, y, z;
        double obsX, obsY, obsRX;


    };

    struct PlaneFittingErrorWithFixedPoints
    {
        PlaneFittingErrorWithFixedPoints(const double _x, const double _y, const double _z, const double _pi0, const double _pi1, const double _pi2, const double _d):
                x(_x), y(_y), z(_z), pi0(_pi0), pi1(_pi1), pi2(_pi2), d(_d){}

        template <typename T>
        bool operator() (const T* const camera, T* residuals) const {
            // camera[0, 1, 2] are the angle-axis rotation
            T p[3];
            T point[3] = {T(x), T(y), T(z)};
            ceres::AngleAxisRotatePoint(camera, point, p);

            // camera[3,4,5] are the translation
            p[0] += camera[3];
            p[1] += camera[4];
            p[2] += camera[5];

            // If the point on the plane
            T err = pi0 * p[0] + pi1 * p[1] + pi2 * p[2] - d;
            residuals[0] = sqrt(err * err);

            return true;
        }

        // Factory to hide the construction of the CostFunction object from the client code.
        static ceres::CostFunction* Create(const double _x, const double _y, const double _z,
                                           const double _pi0, const double _pi1, const double _pi2, const double _d)
        {
            return (new ceres::AutoDiffCostFunction<PlaneFittingErrorWithFixedPoints, 1, 6>(new PlaneFittingErrorWithFixedPoints(_x, _y, _z, _pi0, _pi1, _pi2, _d)));
        }

        double x, y, z;
        double pi0, pi1, pi2, d;
    };

    struct PlaneFittingError
    {
        PlaneFittingError(const double _pi0, const double _pi1, const double _pi2, const double _d):
                pi0(_pi0), pi1(_pi1), pi2(_pi2), d(_d){}

        template <typename T>
        bool operator() (const T* camera,
                const T* const point,
        T* residuals) const {
            // camera[0, 1, 2] are the angle-axis rotation
            T p[3];
            ceres::AngleAxisRotatePoint(camera, point, p);

            // camera[3,4,5] are the translation
            p[0] += camera[3];
            p[1] += camera[4];
            p[2] += camera[5];

            // If the point on the plane
            T err = pi0 * p[0] + pi1 * p[1] + pi2 * p[2] - d;
            residuals[0] = sqrt(err * err);

            return true;
        }

        // Factory to hide the construction of the CostFunction object from the client code.
        static ceres::CostFunction* Create(const double _pi0, const double _pi1, const double _pi2, const double _d)
        {
            return (new ceres::AutoDiffCostFunction<PlaneFittingError, 1, 6, 3>(new PlaneFittingError(_pi0, _pi1, _pi2, _d)));
        }
        double pi0, pi1, pi2, d;
    };

    struct PointsAndPlaneError
    {
        PointsAndPlaneError(float _obsX, float _obsY, float _obsRX, float _pi0, double _pi1, double _pi2, double _d){}
        template <typename T>
        bool operator() (const T* const camera,
                         const T* const point,
                         T* residuals) const {
            // camera[0, 1, 2] are the angle-axis rotation
            T p[3];
            ceres::AngleAxisRotatePoint(camera, point, p);

            // camera[3,4,5] are the translation
            p[0] += camera[3];
            p[1] += camera[4];
            p[2] += camera[5];

            const T& fx = camera[6];
            const T& fy = camera[7];
            const T& cx = camera[8];
            const T& cy = camera[9];
            const T& bf = camera[10];

            const T& invz = 1.0 / p[2];
            const T& u = fx * p[0] * invz + cx;
            const T& v = fy * p[1] * invz + cy;
            const T& dis = bf / p[2];

            residuals[0] = sqrt((u - T(obsX)) * (u - T(obsX)));
            residuals[1] = sqrt((v - T(obsY)) * (v - T(obsY)));
            residuals[2] = sqrt((dis - T(obsX - obsRX)) * (dis - T(obsX - obsRX)));

            // If the point on the plane
//            T err = pi0 * p[0] + pi1 * p[1] + pi2 * p[2] - d;
//            residuals[3] = sqrt(err * err);
            // residuals[3] = T(pi0 * p[0] + pi1 * p[1] + pi2 * p[2] - d);

            return true;
        }
        // Factory to hide the construction of the CostFunction object from the client code.
        static ceres::CostFunction* Create(float _obsX, float _obsY, float _obsRX, const double _pi0, const double _pi1, const double _pi2, const double _d)
        {
            return (new ceres::AutoDiffCostFunction<PointsAndPlaneError, 3, 6, 3>(new PointsAndPlaneError(_obsX, _obsY, _obsRX, _pi0, _pi1, _pi2, _d)));
        }

        double obsX, obsY, obsRX, pi0, pi1, pi2, d;
    };


    class Optimizer
    {
    public:
        void static LocalBundleAdjustment(KeyFrame *pKF);

        int static PoseOptimization(Frame *pFrame);

        int static PoseOptimizationWidthFixedPoints(Frame* pFrame);

        int static PoseOptimizationBasedOnPlane(Frame* pFrame);

        void static PoseOptimizationByCombiningPointsAndPlanes(Frame* pFrame);

        void static LocalBundleAdjustmentByCombiningPointsAndPlanes(KeyFrame* pKF);
    };
}

#endif //EPIPOLARPLANEMATCH_OPTIMIZER_H
