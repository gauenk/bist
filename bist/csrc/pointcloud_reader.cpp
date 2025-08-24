#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>

#include "pointcloud_reader.h"

bool PointCloudReader::readOFFPoints(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return false;
    }
    
    std::string line;
    
    // Read header
    std::getline(file, line);
    if (line != "OFF") {
        std::cerr << "Error: Not a valid OFF file" << std::endl;
        return false;
    }
    
    // Read counts
    int n_vertices, n_faces, n_edges;
    file >> n_vertices >> n_faces >> n_edges;
    
    // Reserve space for points
    points.reserve(n_vertices);
    
    // Read only vertices (skip faces entirely)
    for (int i = 0; i < n_vertices; i++) {
        Point3D p;
        file >> p.x >> p.y >> p.z;
        points.push_back(p);
    }
    
    // No need to read faces - we're done!
    file.close();
    return true;
}

// Normalize to unit sphere
void PointCloudReader::normalize() {
    if (points.empty()) return;
    
    // Find bounding box
    float min_x = points[0].x, max_x = points[0].x;
    float min_y = points[0].y, max_y = points[0].y;
    float min_z = points[0].z, max_z = points[0].z;
    
    for (const auto& p : points) {
        min_x = std::min(min_x, p.x);
        max_x = std::max(max_x, p.x);
        min_y = std::min(min_y, p.y);
        max_y = std::max(max_y, p.y);
        min_z = std::min(min_z, p.z);
        max_z = std::max(max_z, p.z);
    }
    
    // Center and scale
    float center_x = (min_x + max_x) / 2.0f;
    float center_y = (min_y + max_y) / 2.0f;
    float center_z = (min_z + max_z) / 2.0f;
    
    float scale = std::max({max_x - min_x, max_y - min_y, max_z - min_z});
    
    for (auto& p : points) {
        p.x = (p.x - center_x) / scale * 2.0f;
        p.y = (p.y - center_y) / scale * 2.0f;
        p.z = (p.z - center_z) / scale * 2.0f;
    }
}
    
// Subsample to fixed number of points
void PointCloudReader::subsample(int target_points) {
    if (points.empty() || target_points >= (int)points.size()) return;
    
    std::vector<Point3D> subsampled;
    subsampled.reserve(target_points);
    
    float step = (float)points.size() / target_points;
    for (int i = 0; i < target_points; i++) {
        int idx = (int)(i * step);
        subsampled.push_back(points[idx]);
    }
    
    points = std::move(subsampled);
}
    
// Get as flat array for ML frameworks
std::vector<float> PointCloudReader::getPointArray() {
    std::vector<float> array;
    array.reserve(points.size() * 3);
    
    for (const auto& p : points) {
        array.push_back(p.x);
        array.push_back(p.y);
        array.push_back(p.z);
    }
    return array;
}
    
// Get as Nx3 matrix format
void PointCloudReader::copyToMatrix(float* matrix) {
    for (size_t i = 0; i < points.size(); i++) {
        matrix[i * 3 + 0] = points[i].x;
        matrix[i * 3 + 1] = points[i].y;
        matrix[i * 3 + 2] = points[i].z;
    }
}

size_t PointCloudReader::size() const { return points.size(); }

void PointCloudReader::printStats() {
    std::cout << "Point cloud size: " << points.size() << std::endl;
}
