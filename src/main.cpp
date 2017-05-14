
#include <SLAM.h>
#include <Visualizer.h>
#include <CommonFunctions.h>

#include <iostream>
#include <vector>

using namespace std;
using namespace EpipolarPlaneMatch;

int main() {

    std::string sequencePath = "/home/feixue/Research/Dataset/dataset/sequences/06";
    // "/home/feixue/Research/Dataset/data/Desktop/images/07";;//
    std::string settingPath =  //"/home/feixue/Research/Code/SLAM/Stereo_SLAM/Examples/Stereo/pku-06.yaml"; //
            "/home/feixue/Research/Code/SLAM/EpipolarPlaneMatch/Examples/KITTI04-12.yaml";//

//    EpipolarPlaneMatch::Visualizer viz;
//    viz.Run();

    EpipolarPlaneMatch::SLAM slam(settingPath);

    std::vector<std::string> vLeftImagePaths;
    std::vector<std::string> vRightImagePaths;
    std::vector<double> vTimeStamps;

    LoadImages(sequencePath, vLeftImagePaths, vRightImagePaths, vTimeStamps);

    // Process images
    int nImages = vLeftImagePaths.size();
    for (int i = 0; i < 10; ++i)
    {
        cv::Mat imLeft = cv::imread(vLeftImagePaths[i], CV_LOAD_IMAGE_UNCHANGED);
        cv::Mat imRight = cv::imread(vRightImagePaths[i], CV_LOAD_IMAGE_UNCHANGED);

        if (imLeft.empty() || imRight.empty())
        {
            LOG(INFO) << "Read image erro!";
            break;
        }

        LOG(INFO) << "Process Frame: " << i << " .......";

        slam.GrabStereoImage(imLeft, imRight, vTimeStamps[i]);

//        cv::imshow("img", imLeft);
//        cv::waitKey(10);

        LOG(INFO) << "Process Frame: " << i << " finished.";
        std::cout << std::endl;
    }

    std::cout << "Hello, World!" << std::endl;
    return 0;
}