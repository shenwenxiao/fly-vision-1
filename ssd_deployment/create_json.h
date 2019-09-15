//
// Created by lswhao6 on 2019/9/8.
//

#ifndef SSD_DEPLOYMENT_CREATE_JSON_H
#define SSD_DEPLOYMENT_CREATE_JSON_H

#endif //SSD_DEPLOYMENT_CREATE_JSON_H

#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <dirent.h>
#include <algorithm>
#include "boxList.h"
#include "nms.h"

bool loadModule(torch::jit::script::Module& module, std::string path);
void getImagesNames(std::vector<std::string>& imageNames, std::string path);
bool getCategories(std::vector<std::string>& categories, std::string path);
torch::Tensor getInput(cv::Mat& image);
at::Tensor computeLocationsPerLevel(int level, at::Tensor& features);
void forwardForSingleFeatureMap(
        std::vector<BoxList>& boxLists,
        std::vector<c10::IValue>& outputList,
        int level,
        int height,
        int width);
BoxList selectOverAllLevels(BoxList& boxList);
void createJson(
        BoxList &boxList,
        std::string& fileName,
        std::vector<std::string>& categories,
        int height,
        int width);

