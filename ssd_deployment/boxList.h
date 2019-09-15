//
// Created by lswhao6 on 2019/9/8.
//

#ifndef SSD_DEPLOYMENT_BOXLIST_H
#define SSD_DEPLOYMENT_BOXLIST_H

#endif //SSD_DEPLOYMENT_BOXLIST_H

#include <torch/script.h>

class BoxList {
public:
    at::Tensor detections;
    at::Tensor scores;
    at::Tensor fields;

public:
    BoxList(at::Tensor detection, at::Tensor scores, at::Tensor fields);
    static BoxList catBoxLists(std::vector<BoxList> &boxLists);
    void clipToImage(int h, int w);
    void removeSmallBox(int minSize);

};