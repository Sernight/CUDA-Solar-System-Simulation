// Mateusz Szmyd 179920
#ifndef PROJECT_H
#define PROJECT_H

#include <string>

#define G       6.6743015e-20       //km^3/(kg*s^2)
#define Gpd     4.7093871384e-10     //km^3/(kg*day^2)              
#define PI      3.14159265358979323846
#define Ms      1
#define PATH    "./data.csv"

__host__ __device__ class Vector3f {
    public:
        float x;
        float y;
        float z;
        __host__ __device__ Vector3f();
        __host__ __device__ ~Vector3f();
        __host__ __device__ Vector3f(float x0, float y0, float z0);
        __host__ __device__ float operator[] (int i);
        __host__ __device__ Vector3f operator+ (Vector3f &b);
        __host__ __device__ Vector3f operator- (Vector3f &b);
        __host__ __device__ Vector3f operator* (float i);
        __host__ __device__ Vector3f operator/ (float i);
};

__host__ __device__ class Point3f {
    public:
        float x;
        float y;
        float z;
        __host__ __device__ Point3f();
        __host__ __device__ ~Point3f();
        __host__ __device__ Point3f(float x0, float y0, float z0);
        __host__ __device__ float operator[] (int i);
        __host__ __device__ Point3f operator+ (Vector3f &v);
        __host__ __device__ Vector3f operator- (Point3f &b);
};

__host__ __device__ float dist(Point3f a, Point3f b);

struct Values {
    Point3f position;   //równoznaczne odległości od środka ciężkości układu
    Vector3f velocity;
};

struct Orb {
    float mass;
    float radius;
    float density;
    Values values;
    float distance;
};

#endif