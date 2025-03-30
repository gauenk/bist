

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include <dirent.h>
#include <unistd.h> // For access()

#include "file_io.h"

// -- opencv --
#include <opencv2/opencv.hpp>


/********************************************



            Read Optical Flow



*********************************************/

std::vector<std::vector<float>> read_flows(const std::string &directory,
                                           int num) {
  std::vector<std::vector<float>> flows;
  for(int frame_index=0; frame_index <= num; frame_index++){
    char _img_name[100];
    sprintf(_img_name, "%05d.flo",frame_index);
    std::string img_name = std::string(_img_name);
    std::string filename = directory + img_name;
    // std::cout << "A: " << filename << std::endl;
    if (fileExists(filename) == 0){ continue; }
    // std::cout << filename << std::endl;
    // cv::Mat flow = readOpticalFlow(filename);
    std::vector<float> flow = readOpticalFlow(filename);
    flows.push_back(flow);
  }
  // exit(1);
  return flows;
}

// cv::Mat readOpticalFlow(const std::string &filename) {
std::vector<float> readOpticalFlow(const std::string &filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open .flo file!" << std::endl;
        exit(-1);
    }

    float tag;
    file.read(reinterpret_cast<char*>(&tag), sizeof(float));
    // printf("tag: %f\n",tag);
    if (tag != 202021.25) {
        std::cerr << "Invalid flow file format!" << std::endl;
        std::vector<float> flow(2);
        // float* flow = nullptr;
        // cv::Mat flow(1,1, CV_32FC2);
        return flow;
    }

    // Read the width and height of the optical flow image
    int width, height;
    file.read(reinterpret_cast<char*>(&width), sizeof(int));
    file.read(reinterpret_cast<char*>(&height), sizeof(int));
    // printf("height,width: %d,%d\n",height,width);

    // printf("WEIRD OFLOW REMOVE ME!\n");
    // width = 480;
    // height = 480;

    // Create a matrix to store the flow data (2 channels: horizontal and vertical flow)
    // cv::Mat flow(height, width, CV_32FC2);
    // float* flow = (float*)malloc(2*height*width*sizeof(float));
    std::vector<float> flow(2*height*width);
    // memset(flow,0,2*height*width*sizeof(float));

    // Read the flow data for each pixel (horizontal and vertical components)
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float fx, fy;
            file.read(reinterpret_cast<char*>(&fx), sizeof(float));
            file.read(reinterpret_cast<char*>(&fy), sizeof(float));
            if (std::isnan(fx) or std::isnan(fy)){
              printf("nan when reading flow: %2.3f, %2.3f\n",fx,fy);
              exit(1);
            }
            // if ((x == 0) and (y== 0)){
            //   printf("fx,fy: %2.3f,%2.3f\n",fx,fy);
            // }
            // flow.at<cv::Vec2f>(y, x) = cv::Vec2f(fx, fy);
            int flow_index = 2*(y * width + x);
            flow[flow_index] = fx;
            flow[flow_index+1] = fy;
        }
    }

    file.close();
    return flow;
}



/********************************************



            Read Image Files



*********************************************/


std::vector<std::string>
get_image_files(const std::string& directory, const std::string& img_ext, bool read_mode){
  std::vector<std::string> files = get_files_in_directory(directory, img_ext, read_mode);
  if (files.size() == 0){
    read_mode = not(read_mode);
    std::string read_mode_s = read_mode ? "video" : "img";
    printf("Trying read mode [%s]\n",read_mode_s.c_str());
    files=get_files_in_directory(directory, img_ext, read_mode);
  }
  return files;
}



std::vector<std::string>
get_files_in_directory(const std::string& directory, const std::string& img_ext, bool mode) {
  if (mode){ // read video
    return get_video_files(directory,img_ext);
  }else{
    return get_images(directory);
  }
}


std::vector<std::string>
get_video_files(const std::string& directory, const std::string& img_ext){
  int nimgs = count_files(directory, img_ext);
  std::vector<std::string> files;

  int sindex = 0; // start index
  for (int frame_index=sindex; frame_index < nimgs+sindex; frame_index++){
    // std::string img_name = std::format("%05d.%s", frame_index, img_ext);
    char _img_name[100];
    sprintf(_img_name, "%05d.%s",frame_index,img_ext.c_str());
    std::string img_name = std::string(_img_name);
    std::string filename = directory + img_name;
    // std::cout << filename <<std::endl;
    // if (not(isImageFile(filename))){ continue; }
    if (fileExists(filename) == 0){
      if (frame_index == 0){ nimgs += 1;}
      continue;
    }
    files.push_back(img_name);
  }
  // std::cout << files.size() <<std::endl;
  // exit(1);

  return files;
}

std::vector<std::string>
get_images(const std::string& directory){
  std::vector<std::string> files;
  DIR* dpdf = opendir(directory.c_str());
  struct dirent *epdf;
  if (dpdf != NULL){
    while (epdf = readdir(dpdf)){
      std::string name =  epdf->d_name;
      if (isImageFile(name)){
        files.push_back(name);
      }
      // if (name == "." || name == ".." || name.length()<=3) {
      //   continue;
      // }
      // std::string lastThree = name.substr(name.length() - 3);
      // if (lastThree != img_ext){ continue; }
      // // printf("%s\n",img_name.c_str());
      // files.push_back(name);
      // // printf("files!\n");
    }
  }
  closedir(dpdf); // Close the directory
  return files;
}


int count_files(const std::string& root, const std::string& img_ext){

  DIR* dir = opendir(root.c_str()); // Open the directory
  if (dir == nullptr) {
    std::cerr << "Error: Could not open directory: " << root << std::endl;
    return -1;
  }

  int count = 0;
  struct dirent* entry;
  while ((entry = readdir(dir)) != nullptr) {
    std::string name = entry->d_name;
    // Skip "." and ".." (current and parent directory entries)
    // if (name == "." || name == ".." || name.length()<=3) {
    //   continue;
    // }
    if (isImageFile(name)){ count++; }
  }

  closedir(dir); // Close the directory

  return count;
}

bool isImageFile(const std::string& name){

  std::vector<std::string> IMG_EXTS = {"png", "jpg", "jpeg"};
  if (name.length()<=3) { return false; }
  std::string lastThree = name.substr(name.length() - 3);

  bool any = false;
  for (std::string img_ext : IMG_EXTS){
    if (lastThree == img_ext){ any = true; }
  }
  return any;
}

bool fileExists(const std::string& path) {
    return access(path.c_str(), F_OK) == 0;
}

