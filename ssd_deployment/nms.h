//
// Created by lswhao6 on 2019/9/9.
//

#ifndef SSD_DEPLOYMENT_NMS_H
#define SSD_DEPLOYMENT_NMS_H

#endif //SSD_DEPLOYMENT_NMS_H

#include <torch/script.h>

at::Tensor nms_cpu(const at::Tensor& dets, const at::Tensor& scores, const float threshold);
