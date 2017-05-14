//
// Created by feixue on 17-5-3.
//

#include <Optimizer.h>
#include <Converter.h>

#include "Thirdparty/g2o/g2o/core/block_solver.h"
#include "Thirdparty/g2o/g2o/core/optimization_algorithm_levenberg.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_eigen.h"
#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"
#include "Thirdparty/g2o/g2o/core/robust_kernel_impl.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_dense.h"
#include "Thirdparty/g2o/g2o/types/types_seven_dof_expmap.h"

#include<Eigen/StdVector>

#include <ceres/cost_function.h>
#include <ceres/loss_function.h>
#include <ceres/problem.h>
#include <ceres/solver.h>
#include <ceres/gradient_problem_solver.h>

using namespace std;

namespace EpipolarPlaneMatch {
    int Optimizer::PoseOptimization(Frame *pFrame) {

        g2o::SparseOptimizer optimizer;
        g2o::BlockSolver_6_3::LinearSolverType *linearSolver;

        linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();
        g2o::BlockSolver_6_3 *solver_ptr = new g2o::BlockSolver_6_3(linearSolver);
        g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
        optimizer.setAlgorithm(solver);

        // Set Frame vertex
        g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));
        vSE3->setId(0);
        vSE3->setFixed(false);
        optimizer.addVertex(vSE3);

        // Set MapPoint vertex
        const int N = pFrame->N;

        std::vector<g2o::EdgeStereoSE3ProjectXYZOnlyPose *> vpEdgesStereo;
        std::vector<size_t> vnIndexEdgeStereo;
        vpEdgesStereo.reserve(N);
        vnIndexEdgeStereo.reserve(N);

        const float deltaStereo = sqrt(7.815);

        int nInitialCorrespondences = 0;
        for (int i = 0; i < N; ++i) {
            MapPoint *pMP = pFrame->mvpMapPoints[i];
            if (pMP) {
                nInitialCorrespondences++;
                pFrame->mvbOutlier[i] = false;

                // Set Edge
                Eigen::Matrix<double, 3, 1> obs;
                const cv::KeyPoint &kp = pFrame->mvKeysLeft[i];
                const float &kp_ru = pFrame->mvuRight[i];
                obs << kp.pt.x, kp.pt.y, kp_ru;

                g2o::EdgeStereoSE3ProjectXYZOnlyPose *e = new g2o::EdgeStereoSE3ProjectXYZOnlyPose();

                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
                e->setMeasurement(obs);
                const float invSigma2 = pFrame->mvInvLevelSigma2[kp.octave];
                Eigen::Matrix3d Info = Eigen::Matrix3d::Identity() * invSigma2;
                e->setInformation(Info);

                g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);
                rk->setDelta(deltaStereo);

                e->fx = pFrame->fx;
                e->fy = pFrame->fy;
                e->cx = pFrame->cx;
                e->cy = pFrame->cy;
                e->bf = pFrame->mbf;
                cv::Mat Xw = pMP->GetWorldPos();
                e->Xw[0] = Xw.at<float>(0);
                e->Xw[1] = Xw.at<float>(1);
                e->Xw[2] = Xw.at<float>(2);

                optimizer.addEdge(e);

                vpEdgesStereo.push_back(e);
                vnIndexEdgeStereo.push_back(i);
            }
        }

        if (nInitialCorrespondences < 3)
            return 0;

        const float chi2Stereo[4] = {7.815, 7.815, 7.815, 7.815};
        const int its[4] = {10, 10, 10, 10};

        int nBad = 0;
        for (size_t it = 0; it < 4; it++) {
            vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));
            optimizer.initializeOptimization(0);
            optimizer.optimize(its[it]);
            nBad = 0;
            for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++) {
                g2o::EdgeStereoSE3ProjectXYZOnlyPose *e = vpEdgesStereo[i];

                const size_t idx = vnIndexEdgeStereo[i];

                if (pFrame->mvbOutlier[idx]) {
                    e->computeError();
                }

                const float chi2 = e->chi2();

                if (chi2 > chi2Stereo[it]) {
                    pFrame->mvbOutlier[idx] = true;
                    e->setLevel(1);
                    nBad++;
                } else {
                    e->setLevel(0);
                    pFrame->mvbOutlier[idx] = false;
                }

                if (it == 2)
                    e->setRobustKernel(0);
            }

            if (optimizer.edges().size() < 10)
                break;
        }

        // Recover optimized pose and return number of inliers
        g2o::VertexSE3Expmap *vSE3_recov = static_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(0));
        g2o::SE3Quat SE3quat_recov = vSE3_recov->estimate();
        cv::Mat pose = Converter::toCvMat(SE3quat_recov);
        pFrame->SetPose(pose);

        return nInitialCorrespondences - nBad;
    }

    int Optimizer::PoseOptimizationWidthFixedPoints(Frame *pFrame) {
        ceres::Problem problem;
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_SCHUR;
        options.max_num_iterations = 50;
        options.minimizer_progress_to_stdout = true;

        double camera[6];
        double angle[3];
        cv::Mat mR;
        pFrame->mTcw.rowRange(0, 3).colRange(0, 3).convertTo(mR, CV_64FC1);

        cv::Mat mt;
        pFrame->mTcw.rowRange(0, 3).col(3).convertTo(mt, CV_64FC1);

        double *dR = mR.ptr<double>();

        double *dt = mt.ptr<double>();

        ceres::RotationMatrixToAngleAxis(dR, angle);
        camera[0] = angle[0];
        camera[1] = angle[1];
        camera[2] = angle[2];
        camera[3] = dt[0];
        camera[4] = dt[1];
        camera[5] = dt[2];

        const float& fx = pFrame->fx;
        const float& fy = pFrame->fy;
        const float& cx = pFrame->cx;
        const float& cy = pFrame->cy;
        const float& bf = pFrame->mbf;

        LOG(INFO) << "camera extrinsic parameters before optimization: " << camera[0] << " "
                  << camera[1] << " " << camera[2] << " "
                  << camera[3] << " " << camera[4] << " " << camera[5];

        std::vector<double *> points;
        int its[4] = {10, 10, 10, 10};

        for (int i = 0; i < pFrame->N; ++i) {
            MapPoint *pMP = pFrame->mvpMapPoints[i];
            if (!pMP || pFrame->mvbOutlier[i]) continue;

            cv::Mat Xw = pMP->GetWorldPos();

            double *point = new double[3];
            point[0] = Xw.at<float>(0);
            point[1] = Xw.at<float>(1);
            point[2] = Xw.at<float>(2);
            points.push_back(point);

            const cv::KeyPoint &kp = pFrame->mvKeysLeft[i];
            const float &u = kp.pt.x;
            const float &v = kp.pt.y;
            const float &ru = pFrame->mvuRight[i];

            ceres::CostFunction *costFunction = ReprojectionErrorWithFixedPoint::Create(fx, fy, cx, cy, bf,
                                                                                        point[0], point[1], point[2],
                                                                                        u, v, ru);
            ceres::LossFunction *lossFunction = new ceres::HuberLoss(5.0);
            problem.AddResidualBlock(costFunction, lossFunction, camera);
        }
        // Solve
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        std::cout << "Final report:\n" << summary.FullReport();

        LOG(INFO) << "camera extrinsic parameters after optimization: " << camera[0] << " "
                  << camera[1] << " " << camera[2] << " "
                  << camera[3] << " " << camera[4] << " " << camera[5];

        angle[0] = camera[0];
        angle[1] = camera[1];
        angle[2] = camera[2];
        dt[0] = camera[3];
        dt[1] = camera[4];
        dt[2] = camera[5];

        ceres::AngleAxisToRotationMatrix(angle, dR);

        cv::Mat mT = (cv::Mat_<float>(4, 4)
                << dR[0], dR[1], dR[2], dt[0],
                dR[3], dR[4], dR[5], dt[1],
                dR[6], dR[7], dR[8], dt[2],
                0, 0, 0, 1);

        pFrame->SetPose(mT);

        return points.size();
    }

    int Optimizer::PoseOptimizationBasedOnPlane(Frame *pFrame) {

        /*
        ceres::Problem problem;
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_SCHUR;
        options.max_num_iterations = 10;
        options.minimizer_progress_to_stdout = true;

        double camera[6];
        double angle[3];
        cv::Mat mR;
        pFrame->mTcw.rowRange(0, 3).colRange(0, 3).convertTo(mR, CV_64FC1);

        cv::Mat mt;
        pFrame->mTcw.rowRange(0, 3).col(3).convertTo(mt, CV_64FC1);

        double *dR = mR.ptr<double>();

        double *dt = mt.ptr<double>();

        ceres::RotationMatrixToAngleAxis(dR, angle);
        camera[0] = angle[0];
        camera[1] = angle[1];
        camera[2] = angle[2];
        camera[3] = dt[0];
        camera[4] = dt[1];
        camera[5] = dt[2];

        LOG(INFO) << "camera before optimization: " << camera[0] << " "
                  << camera[1] << " " << camera[2] << " "
                  << camera[3] << " " << camera[4] << " " << camera[5];

        std::vector<double *> points;

        for (int i = 0; i < pFrame->N; ++i) {
            MapPoint *pMP = pFrame->mvpMapPoints[i];
            if (!pMP || pFrame->mvbOutlier[i]) continue;

            Plane *pPlane = pFrame->mvpPlanes[i];
            cv::Mat normal = pPlane->GetRawNormal();
            float d = pPlane->GetRawDistance();

            cv::Mat Xw = pMP->GetWorldPos();

            double *point = new double[3];
            point[0] = Xw.at<float>(0);
            point[1] = Xw.at<float>(1);
            point[2] = Xw.at<float>(2);
            points.push_back(point);

//            ceres::CostFunction *costFunction = PlaneFittingError::Create(normal.at<float>(0), normal.at<float>(1),
//                                                                          normal.at<float>(2), d);
            ceres::CostFunction* costFunction = PlaneFittingErrorWithFixedPoints::Create(point[0], point[1], point[2], normal.at<float>(0), normal.at<float>(1),
                                                                          normal.at<float>(2), d);
            ceres::LossFunction *lossFunction = new ceres::HuberLoss(1.0);
            problem.AddResidualBlock(costFunction, lossFunction, camera, point);
        }
        // Solve
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        std::cout << "Final report:\n" << summary.FullReport();

        LOG(INFO) << "camera after optimization: " << camera[0] << " "
                  << camera[1] << " " << camera[2] << " "
                  << camera[3] << " " << camera[4] << " " << camera[5];

        angle[0] = camera[0];
        angle[1] = camera[1];
        angle[2] = camera[2];
        dt[0] = camera[3];
        dt[1] = camera[4];
        dt[2] = camera[5];

        ceres::AngleAxisToRotationMatrix(angle, dR);

        cv::Mat mT = (cv::Mat_<float>(4, 4)
                << dR[0], dR[1], dR[2], dt[0],
                dR[3], dR[4], dR[5], dt[1],
                dR[6], dR[7], dR[8], dt[2],
                0, 0, 0, 1);

        pFrame->SetPose(mT);
        */



        g2o::SparseOptimizer optimizer;
        g2o::BlockSolver_6_3::LinearSolverType *linearSolver;

        linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();
        g2o::BlockSolver_6_3 *solver_ptr = new g2o::BlockSolver_6_3(linearSolver);
        g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
        optimizer.setAlgorithm(solver);

        // Set Frame vertex
        g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));
        vSE3->setId(0);
        vSE3->setFixed(false);
        optimizer.addVertex(vSE3);

        // Set MapPoint vertex
        const int N = pFrame->N;

        std::vector<g2o::EdgeStereoSE3ProjectXYZOnlyPose *> vpEdgesStereo;
        std::vector<size_t> vnIndexEdgeStereo;
        vpEdgesStereo.reserve(N);
        vnIndexEdgeStereo.reserve(N);

        const float deltaStereo = sqrt(7.815);

        int nInitialCorrespondences = 0;
        for (int i = 0; i < N; ++i) {
            MapPoint *pMP = pFrame->mvpMapPoints[i];
            // if (pFrame->mvbOutlier[i]) continue;

            if (pMP) {
                nInitialCorrespondences++;
                pFrame->mvbOutlier[i] = false;

                // Set Edge
                Eigen::Matrix<double, 3, 1> obs;
                const cv::KeyPoint &kp = pFrame->mvKeysLeft[i];
                const float &kp_ru = pFrame->mvuRight[i];
                obs << kp.pt.x, kp.pt.y, kp_ru;

                g2o::EdgeStereoSE3ProjectXYZOnlyPose *e = new g2o::EdgeStereoSE3ProjectXYZOnlyPose();

                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
                e->setMeasurement(obs);
                const float invSigma2 = pFrame->mvInvLevelSigma2[kp.octave];
                Eigen::Matrix3d Info = Eigen::Matrix3d::Identity() * invSigma2;
                e->setInformation(Info);

                g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);
                rk->setDelta(deltaStereo);

                e->fx = pFrame->fx;
                e->fy = pFrame->fy;
                e->cx = pFrame->cx;
                e->cy = pFrame->cy;
                e->bf = pFrame->mbf;
                cv::Mat Xw = pMP->GetWorldPos();
                e->Xw[0] = Xw.at<float>(0);
                e->Xw[1] = Xw.at<float>(1);
                e->Xw[2] = Xw.at<float>(2);

                optimizer.addEdge(e);

                vpEdgesStereo.push_back(e);
                vnIndexEdgeStereo.push_back(i);
            }
        }

        if (nInitialCorrespondences < 3)
            return 0;

        const float chi2Stereo[4] = {7.815, 7.815, 7.815, 7.815};
        const int its[4] = {10, 10, 10, 10};

        int nBad = 0;
        for (size_t it = 0; it < 4; it++) {
            vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));
            optimizer.initializeOptimization(0);
            optimizer.optimize(its[it]);
            nBad = 0;
            for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++) {
                g2o::EdgeStereoSE3ProjectXYZOnlyPose *e = vpEdgesStereo[i];

                const size_t idx = vnIndexEdgeStereo[i];

                if (pFrame->mvbOutlier[idx]) {
                    e->computeError();
                }

                const float chi2 = e->chi2();

                if (chi2 > chi2Stereo[it]) {
                    pFrame->mvbOutlier[idx] = true;
                    e->setLevel(1);
                    nBad++;
                } else {
                    e->setLevel(0);
                    pFrame->mvbOutlier[idx] = false;
                }

                if (it == 2)
                    e->setRobustKernel(0);
            }

            if (optimizer.edges().size() < 10)
                break;
        }

        // Recover optimized pose and return number of inliers
        g2o::VertexSE3Expmap *vSE3_recov = static_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(0));
        g2o::SE3Quat SE3quat_recov = vSE3_recov->estimate();
        cv::Mat pose = Converter::toCvMat(SE3quat_recov);
        pFrame->SetPose(pose);

        return nInitialCorrespondences - nBad;
    }

    void Optimizer::PoseOptimizationByCombiningPointsAndPlanes(Frame *pFrame) {
        ceres::Problem problem;
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_SCHUR;
        options.max_num_iterations = 10;
        options.minimizer_progress_to_stdout = true;

        double camera[11];
        double angle[3];
        cv::Mat mR;
        pFrame->mTcw.rowRange(0, 3).colRange(0, 3).convertTo(mR, CV_64FC1);

        cv::Mat mt;
        pFrame->mTcw.rowRange(0, 3).col(3).convertTo(mt, CV_64FC1);

        double *dR = mR.ptr<double>();

        double *dt = mt.ptr<double>();

        ceres::RotationMatrixToAngleAxis(dR, angle);
        camera[0] = angle[0];
        camera[1] = angle[1];
        camera[2] = angle[2];
        camera[3] = dt[0];
        camera[4] = dt[1];
        camera[5] = dt[2];
        camera[6] = pFrame->fx;
        camera[7] = pFrame->fy;
        camera[8] = pFrame->cx;
        camera[9] = pFrame->cy;
        camera[10] = pFrame->mbf;

        LOG(INFO) << "camera extrinsic parameters before optimization: " << camera[0] << " "
                  << camera[1] << " " << camera[2] << " "
                  << camera[3] << " " << camera[4] << " " << camera[5];
        LOG(INFO) << "camera intrinsic parameters before optimization: " << camera[6] << " "
                  << camera[7] << " " << camera[8] << " "
                  << camera[9] << " " << camera[10];

        std::vector<double *> points;

        for (int i = 0; i < pFrame->N; ++i) {
            MapPoint *pMP = pFrame->mvpMapPoints[i];
            if (!pMP || pFrame->mvbOutlier[i]) continue;

            Plane *pPlane = pFrame->mvpPlanes[i];
            cv::Mat normal = pPlane->GetRawNormal();
            float d = pPlane->GetRawDistance();

            cv::Mat Xw = pMP->GetWorldPos();

            double *point = new double[3];
            point[0] = Xw.at<float>(0);
            point[1] = Xw.at<float>(1);
            point[2] = Xw.at<float>(2);
            points.push_back(point);

            const cv::KeyPoint &kp = pFrame->mvKeysLeft[i];
            const float &u = kp.pt.x;
            const float &v = kp.pt.y;
            const float &ru = pFrame->mvuRight[i];

            ceres::CostFunction *costFunction = PointsAndPlaneError::Create(u, v, ru, normal.at<float>(0),
                                                                            normal.at<float>(1), normal.at<float>(2),
                                                                            d);
            ceres::LossFunction *lossFunction = new ceres::HuberLoss(1.0);
            problem.AddResidualBlock(costFunction, lossFunction, camera, point);
        }
        // Solve
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        std::cout << "Final report:\n" << summary.FullReport();

        LOG(INFO) << "camera extrinsic parameters after optimization: " << camera[0] << " "
                  << camera[1] << " " << camera[2] << " "
                  << camera[3] << " " << camera[4] << " " << camera[5];
        LOG(INFO) << "camera intrinsic parameters after optimization: " << camera[6] << " "
                  << camera[7] << " " << camera[8] << " "
                  << camera[9] << " " << camera[10];

        angle[0] = camera[0];
        angle[1] = camera[1];
        angle[2] = camera[2];
        dt[0] = camera[3];
        dt[1] = camera[4];
        dt[2] = camera[5];

        ceres::AngleAxisToRotationMatrix(angle, dR);

        cv::Mat mT = (cv::Mat_<float>(4, 4)
                << dR[0], dR[1], dR[2], dt[0],
                dR[3], dR[4], dR[5], dt[1],
                dR[6], dR[7], dR[8], dt[2],
                0, 0, 0, 1);

        pFrame->SetPose(mT);
    }

    void Optimizer::LocalBundleAdjustment(KeyFrame *pKF) {
        std::list<KeyFrame *> lLocalKeyFrames;

        lLocalKeyFrames.push_back(pKF);
        pKF->mnBALocalKF = pKF->mnId;

        const std::vector<KeyFrame *> vNeighKFs = pKF->GetVectorCovisibleKeyFrames();
        for (int i = 0, iend = vNeighKFs.size(); i < iend; ++i) {
            KeyFrame *pKFi = vNeighKFs[i];
            pKFi->mnBALocalKF = pKF->mnId;
            lLocalKeyFrames.push_back(pKFi);
        }

        // Local MapPoints seen in local KeyFrames
        std::list<MapPoint *> lLocalMapPoints;
        for (std::list<KeyFrame *>::iterator lit = lLocalKeyFrames.begin(), lend = lLocalKeyFrames.end();
             lit != lend; lit++) {
            std::vector<MapPoint *> vpMPs = (*lit)->GetMapPointMatches();
            for (std::vector<MapPoint *>::iterator vit = vpMPs.begin(), vend = vpMPs.end(); vit != vend; vit++) {
                MapPoint *pMP = *vit;
                if (pMP)
                    if (pMP->mnBALocalKF != pKF->mnId) {
                        pMP->mnBALocalKF = pKF->mnId;
                        lLocalMapPoints.push_back(pMP);
                    }
            }
        }

        // Fixed KeyFrames, KeyFrames that see Local MapPoints but that are not Local KeyFrames
        std::list<KeyFrame *> lFixedCameras;
        for (std::list<MapPoint *>::iterator lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end();
             lit != lend; ++lit) {
            std::map<KeyFrame *, size_t> obs = (*lit)->GetAllObservations();
            for (std::map<KeyFrame *, size_t>::iterator mit = obs.begin(), mend = obs.end(); mit != mend; mit++) {
                KeyFrame *pKFi = mit->first;

                if (pKFi->mnBALocalKF != pKF->mnId && pKFi->mnBAFixedKF != pKF->mnId) {
                    pKFi->mnBAFixedKF = pKF->mnId;
                    lFixedCameras.push_back(pKFi);
                }
            }
        }


        // Setup optimizer
        g2o::SparseOptimizer optimizer;
        g2o::BlockSolver_6_3::LinearSolverType *linearSolver;

        linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();

        g2o::BlockSolver_6_3 *solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

        g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
        optimizer.setAlgorithm(solver);

        unsigned long maxKFid = 0;

        // Set Local KeyFrame vertices
        for (std::list<KeyFrame *>::iterator lit = lLocalKeyFrames.begin(), lend = lLocalKeyFrames.end();
             lit != lend; lit++) {
            KeyFrame *pKFi = *lit;
            g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();
            vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
            vSE3->setId(pKFi->mnId);
            vSE3->setFixed(pKFi->mnId == 0);
            optimizer.addVertex(vSE3);
            if (pKFi->mnId > maxKFid)
                maxKFid = pKFi->mnId;
        }

        // Set Fixed KeyFrame vertices
        for (std::list<KeyFrame *>::iterator lit = lFixedCameras.begin(), lend = lFixedCameras.end();
             lit != lend; lit++) {
            KeyFrame *pKFi = *lit;
            g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();
            vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
            vSE3->setId(pKFi->mnId);
            vSE3->setFixed(true);
            optimizer.addVertex(vSE3);
            if (pKFi->mnId > maxKFid)
                maxKFid = pKFi->mnId;
        }

        LOG(INFO) << "In Local BA Cameras no fixed: " << lLocalKeyFrames.size() << " , fixed: " << lFixedCameras.size();


        // Set MapPoint vertices
        const int nExpectedSize = (lLocalKeyFrames.size() + lFixedCameras.size()) * lLocalMapPoints.size();
        vector<g2o::EdgeStereoSE3ProjectXYZ *> vpEdgesStereo;
        vpEdgesStereo.reserve(nExpectedSize);

        vector<KeyFrame *> vpEdgeKFStereo;
        vpEdgeKFStereo.reserve(nExpectedSize);

        vector<MapPoint *> vpMapPointEdgeStereo;
        vpMapPointEdgeStereo.reserve(nExpectedSize);

        const float thHuberStereo = sqrt(7.815);

        for (list<MapPoint *>::iterator lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end();
             lit != lend; lit++)
        {
            MapPoint *pMP = *lit;
            g2o::VertexSBAPointXYZ *vPoint = new g2o::VertexSBAPointXYZ();
            vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));
            int id = pMP->mnId + maxKFid + 1;
            vPoint->setId(id);
            vPoint->setMarginalized(true);
            optimizer.addVertex(vPoint);

            const map<KeyFrame *, size_t> observations = pMP->GetAllObservations();

            //Set edges
            for (map<KeyFrame *, size_t>::const_iterator mit = observations.begin(), mend = observations.end();
                 mit != mend; mit++) {
                KeyFrame *pKFi = mit->first;

                const cv::KeyPoint &kpUn = pKFi->mvKeys[mit->second];

                // Monocular observation
                if (pKFi->mvuRight[mit->second] >= 0) {
                    Eigen::Matrix<double, 3, 1> obs;
                    const float kp_ur = pKFi->mvuRight[mit->second];
                    obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                    g2o::EdgeStereoSE3ProjectXYZ *e = new g2o::EdgeStereoSE3ProjectXYZ();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKFi->mnId)));
                    e->setMeasurement(obs);
                    const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                    Eigen::Matrix3d Info = Eigen::Matrix3d::Identity() * invSigma2;
                    e->setInformation(Info);

                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberStereo);

                    e->fx = pKFi->fx;
                    e->fy = pKFi->fy;
                    e->cx = pKFi->cx;
                    e->cy = pKFi->cy;
                    e->bf = pKFi->mbf;

                    optimizer.addEdge(e);
                    vpEdgesStereo.push_back(e);
                    vpEdgeKFStereo.push_back(pKFi);
                    vpMapPointEdgeStereo.push_back(pMP);
                }
                //}
            }
        }

        optimizer.initializeOptimization();
        optimizer.optimize(5);

        bool bDoMore = true;

        if (bDoMore) {
            for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++) {
                g2o::EdgeStereoSE3ProjectXYZ *e = vpEdgesStereo[i];
                MapPoint *pMP = vpMapPointEdgeStereo[i];

                if (e->chi2() > 7.815 || !e->isDepthPositive()) {
                    e->setLevel(1);
                }

                e->setRobustKernel(0);
            }

            // Optimize again without the outliers
            optimizer.initializeOptimization(0);
            optimizer.optimize(10);

        }

        vector<pair<KeyFrame *, MapPoint *> > vToErase;
        vToErase.reserve(vpEdgesStereo.size());

        for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++) {
            g2o::EdgeStereoSE3ProjectXYZ *e = vpEdgesStereo[i];
            MapPoint *pMP = vpMapPointEdgeStereo[i];

            if (e->chi2() > 7.815 || !e->isDepthPositive()) {
                KeyFrame *pKFi = vpEdgeKFStereo[i];
                vToErase.push_back(make_pair(pKFi, pMP));
            }
        }

        if (!vToErase.empty()) {
            for (size_t i = 0; i < vToErase.size(); i++) {
                KeyFrame *pKFi = vToErase[i].first;
                MapPoint *pMPi = vToErase[i].second;
                pKFi->EraseMapPointMatch(pMPi);
                pMPi->EraseObservation(pKFi);
            }
        }

        // Recover optimized data

        //Keyframes
        for (list<KeyFrame *>::iterator lit = lLocalKeyFrames.begin(), lend = lLocalKeyFrames.end();
             lit != lend; lit++) {
            KeyFrame *pKF = *lit;
            g2o::VertexSE3Expmap *vSE3 = static_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(pKF->mnId));
            g2o::SE3Quat SE3quat = vSE3->estimate();
            pKF->SetPose(Converter::toCvMat(SE3quat));
        }

        //Points
        for (list<MapPoint *>::iterator lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end();
             lit != lend; lit++) {
            MapPoint *pMP = *lit;
            g2o::VertexSBAPointXYZ *vPoint = static_cast<g2o::VertexSBAPointXYZ *>(optimizer.vertex(
                    pMP->mnId + maxKFid + 1));
            pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
            pMP->UpdateNormalAndDepth();
        }

    }

    void Optimizer::LocalBundleAdjustmentByCombiningPointsAndPlanes(KeyFrame *pKF) {
    }
}