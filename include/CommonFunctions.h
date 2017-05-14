//
// Created by feixue on 17-5-1.
//

#ifndef EPIPOLARPLANEMATCH_COMMONFUNCTIONS_H
#define EPIPOLARPLANEMATCH_COMMONFUNCTIONS_H

#include <opencv2/opencv.hpp>

#include <glog/logging.h>
#include <vector>
#include <unistd.h>
#include <dirent.h>

using namespace std;

inline void LoadImages(const std::string& filePath, std::vector<std::string>& leftImagePaths, std::vector<std::string>& rightImagePath, std::vector<double>& vTimeStamps)
{
    // Load timestamps
    ifstream fTimes;
    string strPathTimeFile = filePath + "/times.txt";
    fTimes.open(strPathTimeFile.c_str());
    while(!fTimes.eof())
    {
        string s;
        getline(fTimes,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            ss >> t;
            vTimeStamps.push_back(t);
        }
    }

    // Load images
    string strPrefixLeft = filePath + "/image_0/";
    string strPrefixRight = filePath + "/image_1/";

    DIR* dir;
    struct dirent* ptr;
    char base[1024];
    if ((dir = opendir(strPrefixLeft.c_str())) == NULL) {
        LOG(INFO) << "Open left image director error.";
        return;
    }
    while((ptr=readdir(dir)) != NULL)
    {
        if ((strcmp(ptr->d_name, ".") == 0) || strcmp(ptr->d_name, "..") == 0)  // Current dir or parent dir
            continue;
        else if (ptr->d_type == 8)   // File
        leftImagePaths.push_back(strPrefixLeft + ptr->d_name);
    }
    closedir(dir);


    if ((dir = opendir(strPrefixRight.c_str())) == NULL) {
        LOG(INFO) << "Open right image director error.";
        return;
    }

    while((ptr=readdir(dir)) != NULL)
    {
        if ((strcmp(ptr->d_name, ".") == 0 || strcmp(ptr->d_name, "..") == 0))  // Current dir or parent dir
            continue;
        else if (ptr->d_type == 8)   // File
            rightImagePath.push_back(strPrefixRight + ptr->d_name);
    }
    closedir(dir);

    sort(leftImagePaths.begin(), leftImagePaths.end());
    sort(rightImagePath.begin(), rightImagePath.end());
}

#endif //EPIPOLARPLANEMATCH_COMMONFUNCTIONS_H
