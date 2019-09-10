//
// Created by lswhao6 on 2019/9/9.
//
#include "boxList.h"

BoxList::BoxList(at::Tensor detections, at::Tensor scores, at::Tensor fields) {
    this->detections = detections;
    this->scores = scores;
    this->fields = fields;
}

BoxList BoxList::catBoxLists(std::vector<BoxList> &boxLists) {
    std::vector<at::Tensor> detectionVec;
    std::vector<at::Tensor> scoresVec;
    std::vector<at::Tensor> fieldsVec;
    for(int i = 0; i < boxLists.size(); i++){
        detectionVec.push_back(boxLists[i].detections);
        scoresVec.push_back(boxLists[i].scores);
        fieldsVec.push_back(boxLists[i].fields);
    }
    at::Tensor detection = at::cat(detectionVec, 0);
    at::Tensor scores = at::cat(scoresVec, 0);
    at::Tensor labels = at::cat(fieldsVec, 0);
    return BoxList(detection, scores, labels);
}

void BoxList::clipToImage(int h, int w) {
    int toRemove = 1;
    auto tensors = this->detections.split(1, 1);
    tensors[0].clamp_(0, w - toRemove);
    tensors[1].clamp_(0, h - toRemove);
    tensors[2].clamp_(0, w - toRemove);
    tensors[3].clamp_(0, h - toRemove);
    this->detections = at::cat_out(this->detections, tensors, 1);
}

void BoxList::removeSmallBox(int minSize) {
    int toRemove = 1;
    auto tensors = this->detections.split(1, 1);
    tensors[2] = tensors[2] - tensors[0] + toRemove;
    tensors[3] = tensors[3] - tensors[1] + toRemove;
    this->detections = at::cat_out(this->detections, tensors, 1);
    tensors = this->detections.unbind(1);
    at::Tensor keep = (tensors[2] >= minSize).eq(tensors[3] >= minSize).nonzero().squeeze();
    auto keepAccessor = keep.accessor<long, 1>();
    int keepAccessorSize = keepAccessor.size(0);
    at::Tensor detections = at::empty({keepAccessorSize, 4});
    at::Tensor scores = at::empty({keepAccessorSize});
    at::Tensor fields = at::empty({keepAccessorSize}, at::ScalarType::Int);
    for(int i = 0; i < keepAccessorSize; i++) {
        scores[i] = this->scores[keepAccessor[i]];
        detections[i] = this->detections[keepAccessor[i]];
        fields[i] = this->fields[keepAccessor[i]];
    }
    this->detections = detections;
    this->scores = scores;
    this->fields = fields;
}
