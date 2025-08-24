
#ifndef POINT_CLOUD_READER_H
#define POINT_CLOUD_READER_H

#include <vector>
#include <string>

struct Point3D {
    float x, y, z;
};

class PointCloudReader {
public:
    std::vector<Point3D> points;
    
    bool readOFFPoints(const std::string& filename);
    void normalize();
    void subsample(int target_points);
    std::vector<float> getPointArray();
    void copyToMatrix(float* matrix);
    size_t size() const;
    void printStats();
};

#endif