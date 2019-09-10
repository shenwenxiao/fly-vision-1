//
// Created by lswhao6 on 2019/9/8.
//

#include "create_json.h"

#define MAX_FILE_NUM 5000
#define LAVEL 5
#define CLASS 80
#define  MAX_DETECTIONS 100
const int PRE_NMS_TOP_N = 1000;
const float MIN_SIZE = 512.0;
const float NMS_TH = 0.6;
const float SCORE_TH = 0.01;
const int BOX_MIN_SIZE = 0;
float pixelMean[] = {0.406, 0.485, 0.456};
float pixelStd[] = {0.225, 0.229, 0.224};
int stride[] = {8, 16, 32, 64, 128};
double fx = 1;

int main() {
    bool flag = false;
    torch::jit::script::Module module;
    flag = loadModule(module, "../model/traced_ssds_mv2_model.pt");
    if(!flag) {
        std::cerr << "Load model error" << "\n";
        return -1;
    }

    std::vector<std::string> imageNames;
    getImagesNames(imageNames, "../testDataset/scaled_images_5000/");
    if (imageNames.size() == 0) {
        std::cerr << "No images found" << "\n";
        return -1;
    }

    std::vector<std::string> categories;
    flag = getCategories(categories, "../coco.txt");
    if(!flag) {
        std::cerr << "Can not open the category file" << "\n";
        return -1;
    }
    if(categories.size() != CLASS){
        std::cerr << "Class number is not " << CLASS << ", please check you model" << "\n";
        return -1;
    }

    std::vector<torch::jit::IValue> inputs;

    torch::Tensor std = torch::from_blob(pixelStd, {3});
    torch::Tensor mean = torch::from_blob(pixelMean, {3});
    torch::jit::IValue outputs;
    std::vector<c10::IValue> outputsList;
    cv::Mat image;
    std::vector<BoxList> boxLists;
    for(int i = 0; i < imageNames.size(); i++) {
		clock_t time0 = clock();
        std::cout << imageNames[i] << "\n";
        image = cv::imread("../testDataset/scaled_images_5000/" + imageNames[i]);
        int oh = image.rows;
        int ow = image.cols;
        torch::Tensor input = getInput(image);
        input -= mean;
        input /= std;
        input = input.permute({0, 3, 1, 2});
        int h = image.rows;
        int w = image.cols;
		clock_t time1 = clock();
		float total0 = (float)(time1 - time0) / CLOCKS_PER_SEC;
		std::cout << "  Time of getting input:" << total0 << "\n";

        // Do inference
        inputs.clear();
        boxLists.clear();
        //inputs.push_back(input.to(at::kCUDA, at::ScalarType::Half));
        inputs.push_back(input.to(at::kCUDA, at::ScalarType::Float));
        outputs = module.forward(inputs);
		clock_t time2 = clock();
		float total1 = (float)(time2 - time1) / CLOCKS_PER_SEC;
		std::cout << "  Time of doing inference:" << total1 << "\n";

		
        outputsList = outputs.toTuple().get()->elements();
        for(int level = 0; level < LAVEL; level++) {
            forwardForSingleFeatureMap(boxLists, outputsList, level, h, w);
        }
        BoxList boxList = BoxList::catBoxLists(boxLists);
        BoxList result = selectOverAllLevels(boxList);
		clock_t time3 = clock();
		float total2 = (float)(time3 - time2) / CLOCKS_PER_SEC;
		std::cout << "  Time of data processing:" << total2 << "\n";

        createJson(result, imageNames[i], categories ,oh, ow);
		clock_t time4 = clock();
		float total3 = (float)(time4 - time3) / CLOCKS_PER_SEC;
		std::cout << "  Time of creating file:" << total3 << "\n";
		float total4 = (float)(time4 - time0) / CLOCKS_PER_SEC;
		std::cout << " Total Time of per frame:" << total4 << "\n";
    }
    return 0;
}

bool loadModule(torch::jit::script::Module& module, std::string path) {
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        module = torch::jit::load(path);
    }
    catch (const c10::Error& e) {
        std::cerr << "Error: Can not load the modle" << std::endl;
        return false;
    }
    std::cout << "Loaded model successfully in "<< path << std::endl;
    //module.to(at::kCUDA, at::ScalarType::Half);
    module.to(at::kCUDA, at::ScalarType::Float);
    module.eval();
    // Create a vector of inputs
    std::vector<torch::jit::IValue> inputs;
    torch::Tensor input = torch::ones({1, 3, 512, 512});
    //inputs.push_back(input.to(at::kCUDA, at::ScalarType::Half));
    inputs.push_back(input.to(at::kCUDA, at::ScalarType::Float));
    module.forward(inputs);
    std::cout << "Inited model successfully" << "\n";
    return true;
}

void getImagesNames(std::vector<std::string>& imagesNames, std::string path) {
    struct dirent *ptr;
    DIR *dir;
    dir = opendir(path.c_str());
    int counter = 0;
    while ((ptr=readdir(dir)) != NULL && counter < MAX_FILE_NUM) {
        if(ptr->d_name[0] == '.')
            continue;
        imagesNames.push_back(ptr->d_name);
        counter++;
    }
    std::sort(imagesNames.begin(), imagesNames.end());
    closedir(dir);
    std::cout << "Got images' names" << "\n";
}

bool getCategories(std::vector<std::string>& categories, std::string path) {
    std::ifstream file(path);
    if(!file.is_open())
        return false;
    std::string category;
    while(std::getline(file, category)) {
        categories.push_back(category);
    }
    return true;
}


torch::Tensor getInput(cv::Mat& image) {
    if(image.rows > image.cols)
		fx = MIN_SIZE / image.cols;
	else
		fx = MIN_SIZE / image.rows;
	clock_t time0 = clock();
	cv::resize(image, image, cv::Size(), fx, fx, cv::INTER_NEAREST);
	clock_t time1 = clock();
	float total0 = (float)(time1 - time0) / CLOCKS_PER_SEC;
	std::cout << "      Time of resizing:" << total0 << "\n";
    int height = image.rows;
    int width = image.cols;
	image.convertTo(image, CV_32FC3, 1 / 255.0);
	torch::Tensor input = torch::empty({1, height, width, 3}, at::ScalarType::Float);
    float *in__ = (float *)input.data_ptr();
	memcpy(in__, image.data, sizeof(float) * 3 * height * width);
	clock_t time2 = clock();
	float total1 = (float)(time2 - time1) / CLOCKS_PER_SEC;
	std::cout << "      Time of transforming:" << total1 << "\n";
	return input;
}

at::Tensor computeLocationsPerLevel(int level, at::Tensor& features) {
    int h = features.size(1);
    int w = features.size(2);
    at::Tensor shiftsX = at::arange(0, w * stride[level], stride[level]);
    at::Tensor shiftsY = at::arange(0, h * stride[level], stride[level]);
    std::vector<at::Tensor> tensorVec;
    tensorVec.clear();
    tensorVec.push_back(shiftsY);
    tensorVec.push_back(shiftsX);
    std::vector<at::Tensor> shifts = at::meshgrid(tensorVec);
    at::Tensor shitfY = shifts[0].reshape({-1});
    at::Tensor shitfX = shifts[1].reshape({-1});
    tensorVec.clear();
    tensorVec.push_back(shitfX);
    tensorVec.push_back(shitfY);
    at::Tensor locations = at::stack(tensorVec, 1) + stride[level] / 2;
    return locations;
}

void forwardForSingleFeatureMap(std::vector<BoxList>& boxLists, std::vector<c10::IValue>& outputList, int level, int h, int w) {
    // put in the same format as locations
    at::Tensor centerness = outputList[level].toTensor().to(at::kCPU, at::ScalarType::Float);
    at::Tensor locations = computeLocationsPerLevel(level, centerness);
    centerness = centerness.reshape({1, -1}).sigmoid();// 1, H*W
    at::Tensor boxRegression = outputList[level + 5].toTensor().to(at::kCPU, at::ScalarType::Float);
    boxRegression = boxRegression.reshape({1, -1, 4});// 1, H*W, 4
    at::Tensor boxCls = outputList[level + 10].toTensor().to(at::kCPU, at::ScalarType::Float);
    int H = boxCls.size(1);
    int W = boxCls.size(2);
    int C = CLASS;
    boxCls = boxCls.reshape({1, -1, C}).sigmoid();// 1, H*W, C

    at::Tensor candidateInds = boxCls > SCORE_TH; // 1, H*W, C
    at::Tensor preNmsTopN = candidateInds.view({1, -1}).sum(1);
    preNmsTopN = preNmsTopN.clamp_max(PRE_NMS_TOP_N);

    // multiply the classification scores with centerness scores
    boxCls = boxCls * centerness.unsqueeze(2);

    at::Tensor perBoxCls = boxCls[0];// H*W, C
    at::Tensor perCandidateInds = candidateInds[0];//H*W, C
    at::Tensor tempPerBoxCls = at::empty({H * W * C});
    int count = 0;
    auto perCandidateIndsAccessor = perCandidateInds.accessor<bool, 2>();
    for (int x = 0; x < H * W; x++) {
        for(int y = 0; y < C; y++)
            if(perCandidateIndsAccessor[x][y])
                tempPerBoxCls[count++] = perBoxCls[x][y];
    }
    perBoxCls = tempPerBoxCls.slice(0, 0, count);

    auto perCandidateNonzeros = perCandidateInds.nonzero().split(1, 1);
    at::Tensor perBoxLoc = perCandidateNonzeros[0].view({-1});
    at::Tensor perClass = (perCandidateNonzeros[1] + 1).view({-1});

    at::Tensor perBoxRegression = at::empty({(H * W * C), 4});
    at::Tensor perLocations = at::empty({(H * W * C), 2});
    std::vector<at::Tensor> perBoxRegressionVec = boxRegression[0].split(1, 0);
    std::vector<at::Tensor> perLocationsVec = locations.split(1, 0);
    auto perBoxLocAccessor = perBoxLoc.accessor<long, 1>();
    int perBoxLocSize = perBoxLoc.size(0);
    for (int x = 0; x < perBoxLocSize; x++) {
        perBoxRegression[x] = perBoxRegressionVec[perBoxLocAccessor[x]].squeeze();
        perLocations[x] = perLocationsVec[perBoxLocAccessor[x]].squeeze();
    }
    perBoxRegression = perBoxRegression.slice(0, 0, perBoxLocSize);
    perLocations = perLocations.slice(0, 0, perBoxLocSize);

    at::Tensor perPreNmsTopN = preNmsTopN[0];

    if (perCandidateInds.sum().item().toInt() > perPreNmsTopN.item().toInt()) {
        at::Tensor topKIndics;
        std::tie(perBoxCls, topKIndics) = perBoxCls.topk(perPreNmsTopN.item().toInt(), -1, true, false);
        auto topKIndicsAccessor = topKIndics.accessor<long, 1>();
        int topKIndicsSize = topKIndics.size(0);
        at::Tensor topKPerClass = at::empty({topKIndicsSize});
        at::Tensor topKPerBoxRegression  = at::empty({topKIndicsSize, 4});
        at::Tensor topKPerLocations = at::empty({topKIndicsSize, 2});
        for (int x = 0; x < topKIndicsSize; x++) {
            topKPerClass[x] = perClass[topKIndicsAccessor[x]];
            topKPerBoxRegression[x] = perBoxRegression[topKIndicsAccessor[x]];
            topKPerLocations[x] = perLocations[topKIndicsAccessor[x]];
        }
        perClass = topKPerClass;
        perBoxRegression = topKPerBoxRegression;
        perLocations = topKPerLocations;
    }

    std::vector<at::Tensor> tensorVec;
    tensorVec.push_back((perLocations.slice(1, 0, 1)
                         - perBoxRegression.slice(1, 0, 1)).reshape({-1}));
    tensorVec.push_back((perLocations.slice(1, 1, 2)
                         - perBoxRegression.slice(1, 1, 2)).reshape({-1}));
    tensorVec.push_back((perLocations.slice(1, 0, 1)
                         + perBoxRegression.slice(1, 2, 3)).reshape({-1}));
    tensorVec.push_back((perLocations.slice(1, 1, 2)
                         + perBoxRegression.slice(1, 3, 4)).reshape({-1}));
    at::Tensor detections = at::stack(tensorVec, 1);
    at::Tensor scores = perBoxCls;
    at::Tensor labels = perClass.to(at::ScalarType::Int);
    BoxList boxList = BoxList(detections, scores, labels);
    //boxList.clipToImage(h, w);
    boxList.removeSmallBox(BOX_MIN_SIZE);
    boxLists.push_back(boxList);
}

BoxList selectOverAllLevels(BoxList& boxlist) {
    std::vector<BoxList> boxListForClass;
    for(int i = 1; i < CLASS + 1; i++) {
        at::Tensor inds = (boxlist.fields == i).nonzero().view({-1});
        auto indsAccessor = inds.accessor<long, 1>();
        int indsAccessorSize = inds.size(0);
//        if(indsAccessorSize == 0)
//            continue;
        at::Tensor scores_i = at::empty({indsAccessorSize});
        at::Tensor detection_i = at::empty({indsAccessorSize, 4});
        for(int j = 0; j < indsAccessorSize; j++) {
            scores_i[j] = boxlist.scores[indsAccessor[j]];
            detection_i[j]  = boxlist.detections[indsAccessor[j]];
        }
        at::Tensor keep;
        if(detection_i.numel() == 0) {
            keep = at::empty({0}, detection_i.options().dtype(at::kLong).device(at::kCPU));
        }
        else {
            keep = nms_cpu(detection_i, scores_i, NMS_TH);
        }
        auto keepAccessor = keep.accessor<long, 1>();
        int keepAccessorSize = keepAccessor.size(0);
        at::Tensor detection_i_keep = at::empty({keepAccessorSize, 4});
        at::Tensor scores_i_keep = at::empty({keepAccessorSize});
        for(int j = 0; j < keepAccessorSize; j++) {
            scores_i_keep[j] = scores_i[keepAccessor[j]];
            detection_i_keep[j] = detection_i[keepAccessor[j]];
        }
        at::Tensor fields_i_keep = at::full({keepAccessorSize}, i);
        boxListForClass.push_back(BoxList(detection_i_keep, scores_i_keep, fields_i_keep));
    }

    BoxList result = BoxList::catBoxLists(boxListForClass);
    int detection_num = result.detections.size(0);
    if(detection_num > MAX_DETECTIONS) {
        auto imageThresh = std::get<0>(torch::kthvalue(result.scores, detection_num - MAX_DETECTIONS + 1));
        at::Tensor keep = result.scores >= imageThresh.item();
        keep = keep.nonzero().squeeze(1);
        auto keepAccessor = keep.accessor<long, 1>();
        int keepAccessorSize = keepAccessor.size(0);
        at::Tensor detection = at::empty({keepAccessorSize, 4});
        at::Tensor scores = at::empty({keepAccessorSize});
        at::Tensor fields = at::empty({keepAccessorSize});
        for(int i = 0; i < keepAccessorSize; i++) {
            scores[i] = result.scores[keepAccessor[i]];
            detection[i] = result.detections[keepAccessor[i]];
            fields[i] = result.fields[keepAccessor[i]];
        }
        result.detections = detection;
        result.scores = scores;
        result.fields = fields;
    }
    return result;
}

void createJson(BoxList &boxList, std::string& fileName, std::vector<std::string>& categories, int height, int width) {
    std::ofstream file;
    std::string path = "../testDataset/annotations/";
    file.open(path + fileName + ".json");
    file << "{\"file_name\": \"" << fileName;
    file << "\", \"height\": " << height;
    file << ", \"width\": " << width;
    file << ", \"annos\": [";
    auto detectionAccessor = boxList.detections.accessor<float, 2>();
    auto scoresAccessor = boxList.scores.accessor<float, 1>();
    boxList.fields = boxList.fields.to(at::ScalarType::Int);
    auto labelsAccessor = boxList.fields.accessor<int, 1>();
    for(int i = 0; i < boxList.detections.size(0); i++) {
        if(i)
            file << ", ";
        float locX = detectionAccessor[i][0] / fx;
        float locY = detectionAccessor[i][1] / fx;
        float h = detectionAccessor[i][2] / fx;
        float w = detectionAccessor[i][3] / fx;
        float area = (detectionAccessor[i][2] * detectionAccessor[i][3]) / fx;
        file << "{\"bbox\": [" << locX << ", ";
        file << locY << ", ";
        file << h << ", ";
        file << w <<"], ";
        file << "\"score\": " << scoresAccessor[i] << ", ";
        file << "\"category_name\": \"" << categories[labelsAccessor[i] - 1] << "\", ";
        file << "\"area\": " << area;
        file << "}";
    }
    file << "]}";
    file.close();
}
