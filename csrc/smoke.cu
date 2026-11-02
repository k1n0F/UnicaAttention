// Smoke test for the build and the loader: y = a·x + y.
extern "C" __global__ void axpy(float a, const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}
