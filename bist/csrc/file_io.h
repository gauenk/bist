#include <string>
#include <vector>

// -- opencv --
#include <opencv2/opencv.hpp>

// -- read flows --
// std::vector<cv::Mat>
// read_flows(const std::string &directory, int num);
// cv::Mat readOpticalFlow(const std::string &filename);
std::vector<std::vector<float>>
read_flows(const std::string &directory, int num);
std::vector<float> readOpticalFlow(const std::string &filename);


// -- read images --
std::vector<std::string>
get_image_files(const std::string& directory, const std::string& img_ext, bool mode);
std::vector<std::string>
get_files_in_directory(const std::string& directory, const std::string& ext, bool mode);
std::vector<std::string>
get_video_files(const std::string& directory, const std::string& ext);
std::vector<std::string>
get_images(const std::string& directory);
bool isImageFile(const std::string& name);
int count_files(const std::string& root, const std::string& ext);
bool fileExists(const std::string& path);
