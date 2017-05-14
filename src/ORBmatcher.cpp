//
// Created by feixue on 17-4-29.
//

#include <ORBmatcher.h>
#include <ORBextractor.h>

#include <cvaux.h>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

using namespace std;

namespace EpipolarPlaneMatch
{
    const int ORBmatcher::TH_HIGH = 100;
    const int ORBmatcher::TH_LOW = 50;
    const int ORBmatcher::HISTO_LENGTH = 30;

    ORBmatcher::ORBmatcher(float nnratio, bool checkOri) : mfNNratio(nnratio), mbCheckOrientation(checkOri) {
    }

    float ORBmatcher::RadiusByViewingCos(const float &viewCos) {
        if (viewCos > 0.998)
            return 2.5;
        else
            return 4.0;
    }

    void ORBmatcher::ComputeThreeMaxima(std::vector<int> *histo, const int L, int &ind1, int &ind2, int &ind3) {
        int max1 = 0;
        int max2 = 0;
        int max3 = 0;

        for (int i = 0; i < L; i++) {
            const int s = histo[i].size();
            if (s > max1) {
                max3 = max2;
                max2 = max1;
                max1 = s;
                ind3 = ind2;
                ind2 = ind1;
                ind1 = i;
            } else if (s > max2) {
                max3 = max2;
                max2 = s;
                ind3 = ind2;
                ind2 = i;
            } else if (s > max3) {
                max3 = s;
                ind3 = i;
            }
        }

        if (max2 < 0.1f * (float) max1) {
            ind2 = -1;
            ind3 = -1;
        } else if (max3 < 0.1f * (float) max1) {
            ind3 = -1;
        }
    }

    int ORBmatcher::DescriptorDistance(const cv::Mat &a, const cv::Mat &b) {
        const int *pa = a.ptr<int32_t>();
        const int *pb = b.ptr<int32_t>();

        int dist = 0;

        for (int i = 0; i < 8; i++, pa++, pb++) {
            unsigned int v = *pa ^*pb;
            v = v - ((v >> 1) & 0x55555555);
            v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
            dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
        }

        return dist;
    }

    int ORBmatcher::LineMatch(const Frame &lastFrame, const Frame &currentFrame, const float &ratio,
                              std::vector<cv::DMatch> &good_matches) {

        const std::vector<cv::KeyPoint>& mvKeys1 = lastFrame.mvKeysLeft;
        const std::vector<cv::KeyPoint>& mvKeys2 = currentFrame.mvKeysLeft;

        const cv::Mat &mDesc1 = lastFrame.mDescriptorsLeft;
        const cv::Mat &mDesc2 = currentFrame.mDescriptorsLeft;



      cv::BFMatcher matcher(cv::BFMatcher::BRUTEFORCE_HAMMINGLUT, true); // (cv::HammingLUT)
        // <cv::HammingLUT> matcher;
        std::vector<cv::DMatch> matches;
        matcher.match(mDesc1, mDesc2, matches);

        double max_dist = 0, min_dist = 1e10;
        for (int i = 0; i < mDesc1.rows; ++i) {
            double dist = matches[i].distance;
            if (dist < min_dist) min_dist = dist;
            if (dist > max_dist) max_dist = dist;
        }

        for (int i = 0; i < mDesc1.rows; ++i) {
            int idx1 = matches[i].queryIdx;
            int idx2 = matches[i].trainIdx;
            int dx = abs(mvKeys1[idx1].pt.x - mvKeys2[idx2].pt.x);
            int dy = abs(mvKeys1[idx1].pt.y - mvKeys2[idx2].pt.y);
//
            if (dx > 100 || dy > 100) continue;
            if (matches[i].distance < ratio * max_dist)
                good_matches.push_back(matches[i]);
        }

        return good_matches.size();
    }

    int ORBmatcher::CircleMatch(const Frame &lastFrame, const Frame &currentFrame, const float &ratio,
                                std::vector<cv::DMatch> &good_matches) {
        const cv::Mat &mDesc1 = lastFrame.mDescriptorsLeft;
        const cv::Mat &mDesc2 = currentFrame.mDescriptorsLeft;
        const std::vector<cv::KeyPoint>& mvKeys1 = lastFrame.mvKeysLeft;
        const std::vector<cv::KeyPoint>& mvKeys2 = currentFrame.mvKeysLeft;
        cv::BFMatcher matcher(cv::BFMatcher::BRUTEFORCE_HAMMINGLUT);
        std::vector<cv::DMatch> matches;
        matcher.match(mDesc1, mDesc2, matches);

        double max_dist = 0, min_dist = 1e10;
        for (int i = 0; i < mDesc1.rows; ++i) {
            double dist = matches[i].distance;
            if (dist < min_dist) min_dist = dist;
            if (dist > max_dist) max_dist = dist;
        }

        for (int i = 0; i < mDesc1.rows; ++i) {
            int idx1 = matches[i].queryIdx;
            int idx2 = matches[i].trainIdx;
            int dx = abs(mvKeys1[idx1].pt.x - mvKeys2[idx2].pt.x);
            int dy = abs(mvKeys1[idx1].pt.y - mvKeys2[idx2].pt.y);
//
            if (dx > 100 || dy > 100) continue;

            if (lastFrame.mnMatches[idx1] < 0) continue;
            if (currentFrame.mnMatches[idx2] < 0) continue;

            if (matches[i].distance < ratio * max_dist)
                good_matches.push_back(matches[i]);
        }

        return good_matches.size();
    }

    void ORBmatcher::DrawMatchesVertical(const cv::Mat &m1, const std::vector<cv::KeyPoint> &points1, const cv::Mat &m2,
                                         const std::vector<cv::KeyPoint> &points2,
                                         const std::vector<cv::DMatch> &matches, cv::Mat &out) {
        cv::Mat m = cv::Mat(m1.rows * 2, m1.cols, m1.type());
        m2.copyTo(m.rowRange(0, m1.rows).colRange(0, m1.cols));
        m1.copyTo(m.rowRange(m1.rows, m.rows).colRange(0, m1.cols));

        for (int i = 0; i < matches.size(); ++i)
        {
            int idx1 = matches[i].queryIdx;
            int idx2 = matches[i].trainIdx;

            cv::Point p1 = points1[idx1].pt;
            cv::Point p2 = points2[idx2].pt;

            p1.y += m1.rows;

            circle(m, p1, 5, cv::Scalar(0, 0, 255));
            circle(m, p2, 5, cv::Scalar(0, 0, 255));
            line(m ,p1, p2, cv::Scalar(0, 255, 0), 1);
        }

        out = m.clone();
    }

    void ORBmatcher::DrawMatchesVertical(const cv::Mat &m1, const std::vector<cv::KeyPoint> &points1, const cv::Mat &m2,
                                         const std::vector<cv::KeyPoint> &points2,
                                         const std::vector<int> &matches, cv::Mat &out) {
        cv::Mat m = cv::Mat(m1.rows, m1.cols * 2, m1.type());
        m1.copyTo(m.rowRange(0, m1.rows).colRange(0, m1.cols));
        m2.copyTo(m.rowRange(0, m.rows).colRange(m1.cols, m.cols));

        for (int i = 0; i < points1.size(); ++i)
        {
            if (matches[i] < 0) continue;
            if (i % 20) continue;

            cv::Point p1 = points1[i].pt;
            cv::Point p2 = points2[matches[i]].pt;

            p2.x += m1.cols;

            circle(m, p1, 5, cv::Scalar(0, 0, 255));
            circle(m, p2, 5, cv::Scalar(0, 0, 255));
            line(m ,p1, p2, cv::Scalar(255, 0, 0), 2);
        }

        out = m.clone();
    }

    void ORBmatcher::DrawCircleMatches(const cv::Mat &m1, const std::vector<cv::KeyPoint> &points1,
                                       const cv::Mat &m2, const std::vector<cv::KeyPoint> &points2,
                                       const cv::Mat &m3, const std::vector<cv::KeyPoint> &points3,
                                       const cv::Mat &m4, const std::vector<cv::KeyPoint> &points4,
                                       const std::vector<int> &matches12,
                                       const std::vector<int> &matches34,
                                       const std::vector<cv::DMatch> &matches13, cv::Mat &out) {
        cv::Mat m = cv::Mat(m1.rows * 2, m1.cols * 2, m1.type());
        m3.copyTo(m.rowRange(0, m1.rows).colRange(0, m1.cols));
        m4.copyTo(m.rowRange(0, m1.rows).colRange(m1.cols, m.cols));
        m1.copyTo(m.rowRange(m1.rows, m.rows).colRange(0, m1.cols));
        m2.copyTo(m.rowRange(m2.rows, m.rows).colRange(m1.cols, m.cols));

        for (int i = 0; i < matches13.size(); ++i)
        {
            int idx1 = matches13[i].queryIdx;
            int idx3 = matches13[i].trainIdx;
            int idx2 = matches12[idx1];
            int idx4 = matches34[idx3];

            cv::Point p1 = points1[idx1].pt;
            cv::Point p2 = points2[idx2].pt;
            cv::Point p3 = points3[idx3].pt;
            cv::Point p4 = points4[idx4].pt;
            if (idx3 % 20) continue;

            p4.x += m1.cols;
            p1.y += m1.rows;
            p2.x += m1.cols;
            p2.y += m1.rows;

            circle(m, p1, 5, cv::Scalar(0, 0, 255));
            circle(m, p2, 5, cv::Scalar(0, 0, 255));
            circle(m, p3, 5, cv::Scalar(0, 0, 255));
            circle(m, p4, 5, cv::Scalar(0, 0, 255));

            line(m ,p1, p2, cv::Scalar(255, 0, 0), 2);
            line(m ,p1, p3, cv::Scalar(0, 255, 0), 2);
            line(m ,p3, p4, cv::Scalar(255, 0, 0), 2);
            line(m ,p2, p4, cv::Scalar(0, 255, 0), 2);
        }

        out = m.clone();


    }

    int ORBmatcher::SearchCircleMatchesByProjection(Frame &CurrentFrame, const Frame &LastFrame, const float th) {
        int nmatches = 0;

        // Rotation Histogram (to check rotation consistency)
        vector<int> rotHist[HISTO_LENGTH];
        for (int i = 0; i < HISTO_LENGTH; i++)
            rotHist[i].reserve(500);
        const float factor = 1.0f / HISTO_LENGTH;

        const cv::Mat Rcw = CurrentFrame.mTcw.rowRange(0, 3).colRange(0, 3);
        const cv::Mat tcw = CurrentFrame.mTcw.rowRange(0, 3).col(3);

        const cv::Mat twc = -Rcw.t() * tcw;

        const cv::Mat Rlw = LastFrame.mTcw.rowRange(0, 3).colRange(0, 3);
        const cv::Mat tlw = LastFrame.mTcw.rowRange(0, 3).col(3);

        const cv::Mat tlc = Rlw * twc + tlw;

        const bool bForward = tlc.at<float>(2) > CurrentFrame.mb;
        const bool bBackward = -tlc.at<float>(2) > CurrentFrame.mb;

        std::vector<int> vLastMatches = LastFrame.mnMatches;
        std::vector<int> vCurrentMatches = CurrentFrame.mnMatches;

        for (int i = 0; i < LastFrame.N; i++) {
            MapPoint *pMP = LastFrame.mvpMapPoints[i];
            if (vLastMatches[i] < 0) continue;          // Circle match: ll-lr

            if (pMP) {

                if (!LastFrame.mvbOutlier[i]) {
                    // Project
                    cv::Mat x3Dw = pMP->GetWorldPos();
                    cv::Mat x3Dc = Rcw * x3Dw + tcw;

                    const float xc = x3Dc.at<float>(0);
                    const float yc = x3Dc.at<float>(1);
                    const float invzc = 1.0 / x3Dc.at<float>(2);

                    if (invzc < 0)
                        continue;

                    float u = CurrentFrame.fx * xc * invzc + CurrentFrame.cx;
                    float v = CurrentFrame.fy * yc * invzc + CurrentFrame.cy;

                    if (u < CurrentFrame.mnMinX || u > CurrentFrame.mnMaxX)
                        continue;
                    if (v < CurrentFrame.mnMinY || v > CurrentFrame.mnMaxY)
                        continue;

                    int nLastOctave = LastFrame.mvKeysLeft[i].octave;

                    // Search in a window. Size depends on scale
                    float radius = th * CurrentFrame.mvScaleFactors[nLastOctave];

                    vector<size_t> vIndices;

                    if (bForward)
                        vIndices = CurrentFrame.GetFeaturesInArea(u, v, radius, nLastOctave);
                    else if (bBackward)
                        vIndices = CurrentFrame.GetFeaturesInArea(u, v, radius, 0, nLastOctave);
                    else
                        vIndices = CurrentFrame.GetFeaturesInArea(u, v, radius, nLastOctave - 1, nLastOctave + 1);

                    if (vIndices.empty())
                        continue;

                    const cv::Mat dMP = pMP->GetDescriptor();

                    const cv::Mat& dLL = LastFrame.mDescriptorsLeft.row(i);
                    const cv::Mat& dLR = LastFrame.mDescriptorsRight.row(vLastMatches[i]);

                    int bestDist = 256;
                    int bestIdx = -1;

                    for (vector<size_t>::const_iterator vit = vIndices.begin(), vend = vIndices.end();
                         vit != vend; vit++) {
                        const size_t idx = *vit;

                        if (vCurrentMatches[idx] < 0) continue;    // Circle match: cl-cr

//                        LOG(INFO) << "idx: " << idx << " and matches: " << vCurrentMatches[idx];
//                        if (CurrentFrame.mvpMapPoints[idx])
//                            if (CurrentFrame.mvpMapPoints[idx]->Observations() > 0)
//                                continue;

                        const cv::Mat& dCL = CurrentFrame.mDescriptorsLeft.row(idx);
                        const cv::Mat& dCR = CurrentFrame.mDescriptorsRight.row(vCurrentMatches[idx]);

                        int distLL = DescriptorDistance(dLL, dCL);
                        int distRR = DescriptorDistance(dLR, dCR);

                        if (distLL > TH_HIGH || distRR > TH_HIGH) continue;   // Circle matches: ll-cl, lr-cr
                        float dist = (distLL + distRR) / 2.0;

                        if (dist < bestDist) {
                            bestDist = dist;
                            bestIdx = idx;
                        }
                    }

                    if (bestDist <= TH_HIGH) {
                        CurrentFrame.mvpMapPoints[bestIdx] = pMP;
                        nmatches++;

                        if (mbCheckOrientation) {
                            float rot = LastFrame.mvKeysLeft[i].angle - CurrentFrame.mvKeysLeft[bestIdx].angle;
                            if (rot < 0.0)
                                rot += 360.0f;
                            int bin = round(rot * factor);
                            if (bin == HISTO_LENGTH)
                                bin = 0;
                            assert(bin >= 0 && bin < HISTO_LENGTH);
                            rotHist[bin].push_back(bestIdx);
                        }
                    }
                }
            }
        }

        //Apply rotation consistency
        if (mbCheckOrientation) {
            int ind1 = -1;
            int ind2 = -1;
            int ind3 = -1;

            ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);

            for (int i = 0; i < HISTO_LENGTH; i++) {
                if (i != ind1 && i != ind2 && i != ind3) {
                    for (size_t j = 0, jend = rotHist[i].size(); j < jend; j++) {
                        CurrentFrame.mvpMapPoints[rotHist[i][j]] = static_cast<MapPoint *>(NULL);
                        nmatches--;
                    }
                }
            }
        }

        return nmatches;
    }

    int ORBmatcher::SearchByProjection(Frame &F, const std::vector<MapPoint *> &vpMapPoints, const float th) {
        int nmatches = 0;

        const bool bFactor = th != 1.0;

        for (size_t iMP = 0; iMP < vpMapPoints.size(); iMP++) {
            MapPoint *pMP = vpMapPoints[iMP];
            if (!pMP->mbTrackInView)
                continue;

            if (pMP->IsBad())
                continue;

            const int &nPredictedLevel = pMP->mnTrackScaleLevel;

            // The size of the window will depend on the viewing direction
            float r = RadiusByViewingCos(pMP->mTrackViewCos);

            if (bFactor)
                r *= th;

            const vector<size_t> vIndices =
                    F.GetFeaturesInArea(pMP->mTrackProjX, pMP->mTrackProjY, r * F.mvScaleFactors[nPredictedLevel],
                                        nPredictedLevel - 1, nPredictedLevel);

            if (vIndices.empty())
                continue;

            const cv::Mat MPdescriptor = pMP->GetDescriptor();

            int bestDist = 256;
            int bestLevel = -1;
            int bestDist2 = 256;
            int bestLevel2 = -1;
            int bestIdx = -1;

            // Get best and second matches with near keypoints
            for (vector<size_t>::const_iterator vit = vIndices.begin(), vend = vIndices.end(); vit != vend; vit++) {
                const size_t idx = *vit;

                if (F.mvpMapPoints[idx])
                    if (F.mvpMapPoints[idx]->Observations() > 0)
                        continue;

                if (F.mvuRight[idx] > 0) {
                    const float er = fabs(pMP->mTrackProjXR - F.mvuRight[idx]);
                    if (er > r * F.mvScaleFactors[nPredictedLevel])
                        continue;
                }

                const cv::Mat &d = F.mDescriptorsLeft.row(idx);

                const int dist = DescriptorDistance(MPdescriptor, d);

                if (dist < bestDist) {
                    bestDist2 = bestDist;
                    bestDist = dist;
                    bestLevel2 = bestLevel;
                    bestLevel = F.mvKeysLeft[idx].octave;
                    bestIdx = idx;
                } else if (dist < bestDist2) {
                    bestLevel2 = F.mvKeysLeft[idx].octave;
                    bestDist2 = dist;
                }
            }

            // Apply ratio to second match (only if best and second are in the same scale level)
            if (bestDist <= TH_HIGH) {
                if (bestLevel == bestLevel2 && bestDist > mfNNratio * bestDist2)
                    continue;

                F.mvpMapPoints[bestIdx] = pMP;
                nmatches++;
            }
        }

        return nmatches;
    }
}


