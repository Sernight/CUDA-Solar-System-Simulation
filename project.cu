// Mateusz Szmyd 179920
#include "project.h"
#include <math.h>
#include <iostream>
#include <vector>
#include <cooperative_groups.h>

using namespace std;

Orb *d_obj;
Orb *d_out;
Vector3f *d_accels;

__host__ __device__ Vector3f::Vector3f() {

};

__host__ __device__ Vector3f::~Vector3f() {

};

__host__ __device__ Vector3f::Vector3f(float x0, float y0, float z0) {
    this->x = x0;
    this->y = y0;
    this->z = z0;
};

__host__ __device__ float Vector3f::operator[](int i) {
    if (i == 0) {
        return this->x;
    };
    if (i == 1) {
        return this->y;
    };
    if (i == 2) {
        return this->z;
    };
};

__host__ __device__ Vector3f Vector3f::operator+ (Vector3f &b) {
    Vector3f vec;
    vec.x = this->x + b.x;
    vec.y = this->y + b.y;
    vec.z = this->z + b.z;
    return vec;
};

__host__ __device__ Vector3f Vector3f::operator- (Vector3f &b) {
    Vector3f vec;
    vec.x = this->x - b.x;
    vec.y = this->y - b.y;
    vec.z = this->z - b.z;
    return vec;
};

__host__ __device__ Vector3f Vector3f::operator* (float i) {
    Vector3f vec;
    vec.x = this->x * i;
    vec.y = this->y * i;
    vec.z = this->z * i;
    return vec;
};
__host__ __device__ Vector3f Vector3f::operator/ (float i) {
    Vector3f vec;
    vec.x = this->x / i;
    vec.y = this->y / i;
    vec.z = this->z / i;
    return vec;
};

__host__ __device__ Point3f::Point3f() {

};

__host__ __device__ Point3f::~Point3f() {

};

__host__ __device__ Point3f::Point3f(float x0, float y0, float z0) {
    this->x = x0;
    this->y = y0;
    this->z = z0;
};

__host__ __device__ float Point3f::operator[] (int i) {
    if (i == 0) {
        return this->x;
    };
    if (i == 1) {
        return this->y;
    };
    if (i == 2) {
        return this->z;
    };
};

__host__ __device__ Point3f Point3f::operator+ (Vector3f &v) {
    Point3f p;
    p.x = this->x + v.x;
    p.y = this->y + v.y;
    p.z = this->z + v.z;
    return p;
};

__host__ __device__ Vector3f Point3f::operator- (Point3f &b) {
    Vector3f vec;
    vec.x = this->x - b.x;
    vec.y = this->y - b.y;
    vec.z = this->z - b.z;
    return vec;
};

__host__ __device__ float dist(Point3f a, Point3f b) {
    return sqrtf(powf(a.x - b.x, 2) + powf(a.y - b.y, 2) + powf(a.z - b.z, 2));
}

__device__ bool collision_check(Orb *obj1, Orb *obj2) {
    float distance = dist(obj1->values.position, obj2->values.position);
    if ((obj1->radius  + obj2->radius)*0.98 > distance) {
        if (obj1->mass > obj2->mass) {
            float mass = obj1->mass + obj2->mass;
            float density = obj1->density * obj1->mass / mass + obj2->density * obj2->mass / mass;
            obj2->mass = 0;
            obj2->radius = 0;
            obj1->density = density;
            obj1->mass = mass;
            obj1->radius = sqrtf(0.75*obj1->density / (PI*obj1->mass));
        }
        return true;
    } else {
        return false;
    }
};

// jeden blok to jest obiekt, dla którego liczone są nowe wartości
// jeden wątek w bloku to jest obiekt, który oddziałuje na obiekt z jednego bloku
extern "C" __global__ void update_values(Orb *obj, float dt, Orb *out, Vector3f *accels) {
    // odległość pomiędzy obiektami
    float distance = dist(obj[blockIdx.x].values.position, obj[threadIdx.x].values.position);
    __shared__ bool collision;
    
    Vector3f direct;
    Vector3f acc;

    // obliczenie przyspieszeń do każdego z obiektów z pominięciem tego samego
    if (blockIdx.x != threadIdx.x) {
        direct = (obj[threadIdx.x].values.position - obj[blockIdx.x].values.position) / distance;

        acc.x = G * direct.x * obj[threadIdx.x].mass / powf(distance, 2);
        acc.y = G * direct.y * obj[threadIdx.x].mass / powf(distance, 2);
        acc.z = G * direct.z * obj[threadIdx.x].mass / powf(distance, 2);

    } else {    // przypadek, gdy to jest ten sam obiekt
        acc = Vector3f(0, 0, 0);
        collision = false;
    };

    // suma wszystkich wektorów przyspieszeń
    atomicAdd(&accels[blockIdx.x].x, acc.x);
    atomicAdd(&accels[blockIdx.x].y, acc.y);
    atomicAdd(&accels[blockIdx.x].z, acc.z);

    __shared__ Point3f new_pos;
    __shared__ Vector3f new_vel;

    // wątek odpowiadający temu samemu obiektowi, wykona obliczenia nowej pozycji oraz prędkości,
    // reszta sprawdzi kolizję na podstawie starych danych
    if (blockIdx.x == threadIdx.x) {
        // nowa pozycja
        new_pos.x = obj[blockIdx.x].values.position.x + obj[blockIdx.x].values.velocity.x * dt;
        new_pos.y = obj[blockIdx.x].values.position.y + obj[blockIdx.x].values.velocity.y * dt;
        new_pos.z = obj[blockIdx.x].values.position.z + obj[blockIdx.x].values.velocity.z * dt;

        // nowa prędkość
        new_vel.x = obj[blockIdx.x].values.velocity.x + accels[blockIdx.x].x * dt;
        new_vel.y = obj[blockIdx.x].values.velocity.y + accels[blockIdx.x].y * dt;
        new_vel.z = obj[blockIdx.x].values.velocity.z + accels[blockIdx.x].z * dt;
    } else {
        collision = collision_check(&obj[blockIdx.x], &obj[threadIdx.x]);
    };
    __syncthreads();
    if (blockIdx.x == threadIdx.x) {
        out[blockIdx.x] = obj[blockIdx.x];
        if (!collision) {
            out[blockIdx.x].values.position = new_pos;
            out[blockIdx.x].values.velocity = new_vel;
        };
    };
};

// skopiowanie danych do device'a
extern "C" void initialize(Orb *obj, int length, float dt) {
    cudaMalloc((void **)&d_out, sizeof(Orb) * length);
    cudaMalloc((void **)&d_obj, sizeof(Orb) * length);
    cudaMalloc((void **)&d_accels, sizeof(Vector3f) * length);

    cudaMemcpy(d_obj, obj, sizeof(Orb) * length, cudaMemcpyHostToDevice);
};

// końcowe uwolnienie pamięci
extern "C" void freeMem() {
    cudaFree(d_obj);
    cudaFree(d_out);
    cudaFree(d_accels);
};

extern "C" void update(Orb *obj, int length, float dt) {
    initialize(obj, length, dt);
    update_values<<<length, length>>>(d_obj, dt, d_out, d_accels);
    cudaDeviceSynchronize();
    cudaMemcpy(obj, d_out, sizeof(Orb) * length, cudaMemcpyDeviceToHost);
    freeMem();
};

int main() {

};

