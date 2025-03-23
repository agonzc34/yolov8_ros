// MIT License
//
// Copyright (c) 2025 Alejandro González Cantón
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "yolo_cpp_ros/yolo/utils.hpp"
#include <algorithm>

namespace yolo_onnx_utils {

float iou(const Box &box1, const Box &box2)
{
    float x1 = std::max(box1.x1, box2.x1);
    float y1 = std::max(box1.y1, box2.y1);
    float x2 = std::min(box1.x2, box2.x2);
    float y2 = std::min(box1.y2, box2.y2);

    float intersection = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    float area1 = (box1.x2 - box1.x1) * (box1.y2 - box1.y1);
    float area2 = (box2.x2 - box2.x1) * (box2.y2 - box2.y1);
    float unionArea = area1 + area2 - intersection;

    return intersection / unionArea;
}

std::vector<struct Box> nms(std::vector<Box> &boxes, float iouThreshold, float confThreshold)
{
    std::vector<Box> detections;

    boxes.erase(std::remove_if(boxes.begin(), boxes.end(), [confThreshold](const Box &a) { return a.score < confThreshold; }), boxes.end());
    std::sort(boxes.begin(), boxes.end(), [](const Box &a, const Box &b) { return a.score > b.score; });

    std::vector<bool> keep(boxes.size(), true);
    for (size_t i = 0; i < boxes.size(); ++i)
    {
        if (!keep[i])
            continue;

        detections.push_back(boxes[i]);
        float x1 = boxes[i].x1;
        float y1 = boxes[i].y1;
        float x2 = boxes[i].x2;
        float y2 = boxes[i].y2;
        float score = boxes[i].score;
        int index = boxes[i].index;

        detections.back().x1 = x1;
        detections.back().y1 = y1;
        detections.back().x2 = x2;
        detections.back().y2 = y2;
        detections.back().score = score;
        detections.back().index = index;


        for (size_t j = i + 1; j < boxes.size(); ++j)
        {
            if (keep[j] && iou(boxes[i], boxes[j]) > iouThreshold)
                keep[j] = false;
        }
    }

    return detections;
}
}