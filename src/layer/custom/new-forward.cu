#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

__constant__ float d_k[3136];
__constant__ int d_B;
__constant__ int d_M; 
__constant__ int d_C; 
__constant__ int d_H; 
__constant__ int d_W; 
__constant__ int d_K;
__constant__ int d_wGrid;
__constant__ int d_tileWidth;

__global__ void naive_conv_forward(
	float *y, const float *x, const float *k, const int B, const int M, 
	const int C, const int H, const int W, const int K
)
{	int H_out = H - K + 1;
    int W_out = W - K + 1;
	
	#define y4d(i3, i2, i1, i0) y[(i3)*d_M*H_out*W_out + (i2)*H_out*W_out + (i1)*W_out + i0]
	#define x4d(i3, i2, i1, i0) x[(i3)*d_C*d_H*d_W + (i2)*d_H*d_W + (i1)*d_W + i0]
	#define k4d(i3, i2, i1, i0) d_k[(i3)*d_C*d_K*d_K + (i2)*d_K*d_K + (i1)*d_K + i0]

	int h = blockIdx.y*8+threadIdx.y;
    int w = blockIdx.x*8+threadIdx.x;
	int m = blockIdx.z*8+threadIdx.z;
    
	if(h<H_out && w<W_out && m<M){
	    for(int b=0; b<B; b++) {         // for each image in the batch 
        	y4d(b, m, h, w) = 0;
              
            for(int c=0; c<C; c++) {     // sum over all input feature maps
            	for(int p=0; p<K; p++) { // KxK filter
                	for(int q=0; q<K; q++) { 
						float k4 = k4d(m, c, p, q);

                  	    y4d(b, m, h, w) += x4d(b, c, h + p, w + q) * k4;        
                   	}
             	}
        	}        
    	}
	}

	#undef y4d
	#undef x4d
	#undef k4d
}


__global__ void constantMemory_conv_forward(float *y, const float *x)
{	int H_out = d_H - d_K + 1;
    int W_out = d_W - d_K + 1;
	
	#define y4d(i3, i2, i1, i0) y[(i3)*d_M*H_out*W_out + (i2)*H_out*W_out + (i1)*W_out + i0]
	#define x4d(i3, i2, i1, i0) x[(i3)*d_C*d_H*d_W + (i2)*d_H*d_W + (i1)*d_W + i0]
	#define d_k4d(i3, i2, i1, i0) d_k[(i3)*d_C*d_K*d_K + (i2)*d_K*d_K + (i1)*d_K + i0]

	int h = blockIdx.y*8+threadIdx.y;
    int w = blockIdx.x*8+threadIdx.x;
	int m = blockIdx.z*8+threadIdx.z;
    
	if(h<H_out && w<W_out && m<d_M){
	    for(int b=0; b<d_B; b++) {         // for each image in the batch 
        	y4d(b, m, h, w) = 0;
              
            for(int c=0; c<d_C; c++) {     // sum over all input feature maps
            	for(int p=0; p<d_K; p++) { // KxK filter
                	for(int q=0; q<d_K; q++) {
						y4d(b, m, h, w) += x4d(b, c, h+p, w+q)*d_k4d(m, c, p, q);        
                   	}
             	}
        	}        
    	}
	}

	#undef y4d
	#undef x4d
	#undef d_k4d
}


__global__ void tiling_conv_forward(float *y, const float *x)
{	int H_out = d_H - d_K + 1;
    int W_out = d_W - d_K + 1;
	
	#define y4d(i3, i2, i1, i0) y[(i3)*d_M*H_out*W_out + (i2)*H_out*W_out + (i1)*W_out + i0]
	#define x4d(i3, i2, i1, i0) x[(i3)*d_C*d_H*d_W + (i2)*d_H*d_W + (i1)*d_W + i0]
	#define d_k4d(i3, i2, i1, i0) d_k[(i3)*d_C*d_K*d_K + (i2)*d_K*d_K + (i1)*d_K + i0]

	int b = blockIdx.z;
    int m = blockIdx.x;
    int h = (blockIdx.y/d_wGrid)*d_tileWidth + threadIdx.y;
    int w = (blockIdx.y%d_wGrid)*d_tileWidth + threadIdx.x;
	
	if(b < d_B && m < d_M && h < H_out && w < W_out) {
		float accumulator = 0;
		
		for(int c = 0; c < d_C; c++) {     // sum over all input feature maps
			for(int p = 0; p < d_K; p++) { // KxK filter
				for(int q = 0; q < d_K; q++) {
					accumulator += x4d(b, c, h+p, w+q)*d_k4d(m, c, p, q);        
				}
			}
		}

		y4d(b, m, h, w) = accumulator;        
	}
			
	#undef y4d
	#undef x4d
	#undef d_k4d
}

__global__ void sharedMemory_tiling_conv_forward(float *__restrict__ y, const float *__restrict__ x)
{	__shared__ float d_bmx[25][25];
	int H_out = d_H - d_K + 1;
    int W_out = d_W - d_K + 1;
	int b = blockIdx.z;
    int m = blockIdx.x;
    int tx = threadIdx.y;
    int ty = threadIdx.x;
	int h = (blockIdx.y/d_wGrid)*d_tileWidth + ty;
    int w = (blockIdx.y%d_wGrid)*d_tileWidth + tx;
		
	#define y4d(i3, i2, i1, i0) y[(i3)*d_M*H_out*W_out + (i2)*H_out*W_out + (i1)*W_out + i0]
	#define x4d(i3, i2, i1, i0) x[(i3)*d_C*d_H*d_W + (i2)*d_H*d_W + (i1)*d_W + i0]
	#define d_k4d(i3, i2, i1, i0) d_k[(i3)*d_C*d_K*d_K + (i2)*d_K*d_K + (i1)*d_K + i0]

	if(h < d_H && w < d_W) {
		float accumulator = 0;
		
		for(int c = 0; c < d_C; c++) {     // sum over all input feature maps
			d_bmx[ty][tx] = x4d(b, c, h, w);
			if(ty < d_K && tx < d_K) {
				d_bmx[ty+d_tileWidth][tx+d_tileWidth] = x4d(b, c, h+d_tileWidth, w+d_tileWidth);
			}	
			if(tx < d_K) {
				d_bmx[ty][tx+d_tileWidth] = x4d(b, c, h, w+d_tileWidth);
			}	
			if(ty < d_K) {
				d_bmx[ty+d_tileWidth][tx] = x4d(b, c, h+d_tileWidth, w);
			}
			__syncthreads();
	
			for(int p = 0; p < d_K; p++) { // KxK filter
				for(int q = 0; q < d_K; q++) {
					accumulator += d_bmx[ty+p][tx+q]*d_k4d(m, c, p, q);        
				}
			}
		}

		if(b < d_B && m < d_M && h < H_out && w < W_out) {
			y4d(b, m, h, w) = accumulator;
		}        
	}
			
	#undef y4d
	#undef x4d
	#undef d_k4d
}


__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_y, const float *host_x, const float *host_k, float **device_y_ptr, float **device_x_ptr, float **device_k_ptr, const int B, const int M, const int C, const int H, const int W, const int K)
{   const int hOut = H-K+1;
    const int wOut = W-K+1;
		
	// Allocate GPU memory
	cudaMalloc((void**)device_y_ptr, B*M*wOut*hOut*sizeof(float));//SIZE MAY BE WRONG
  	cudaMalloc((void**)device_x_ptr, B*C*W*H*sizeof(float));
	
	// Copy Global memory to the GPU here
	cudaMemcpy(*device_y_ptr, host_y, B*M*wOut*hOut*sizeof(float), cudaMemcpyHostToDevice);//SIZE MAY BE WRONG
  	cudaMemcpy(*device_x_ptr, host_x, B*C*W*H*sizeof(float), cudaMemcpyHostToDevice);
	
	// Copy Constant memory to the GPU here
	cudaMemcpyToSymbol(d_k, host_k, K*K*C*M*sizeof(float));
	cudaMemcpyToSymbol(d_B, &B, sizeof(int));
  	cudaMemcpyToSymbol(d_M, &M, sizeof(int));
  	cudaMemcpyToSymbol(d_C, &C, sizeof(int));
  	cudaMemcpyToSymbol(d_H, &H, sizeof(int));
  	cudaMemcpyToSymbol(d_W, &W, sizeof(int));
  	cudaMemcpyToSymbol(d_K, &K, sizeof(int));
	
	// Useful snippet for error checking
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){	
	    std::cout<<"CUDA error Transfer: "<<cudaGetErrorString(error)<<std::endl;
  	    exit(-1);
    }
}
 

__host__ void GPUInterface::conv_forward_gpu(float *device_y, const float *device_x, const float *device_k, const int B, const int M, const int C, const int H, const int W, const int K)
{	const int hOut = H - K + 1;
    const int wOut = W - K + 1;
	const int tileWidth = 16;
	
	// Set the Tiling kernel dimensions and call the kernel
  	const int wGrid = ceil(wOut/float(tileWidth));
	const int hGrid = ceil(hOut/float(tileWidth));
	const int Y = wGrid*hGrid;
	cudaMemcpyToSymbol(d_wGrid, &wGrid, sizeof(int));
	cudaMemcpyToSymbol(d_tileWidth, &tileWidth, sizeof(int));
	dim3 block_size(tileWidth, tileWidth, 1); 
  	dim3 grid_size(M, Y, B);
	
	sharedMemory_tiling_conv_forward<<< grid_size, block_size >>>(device_y, device_x);
	cudaDeviceSynchronize();
	
	// Useful snippet for error checking
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){	
	    std::cout<<"CUDA error Launch: "<<cudaGetErrorString(error)<<std::endl;
  	    exit(-1);
    }
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_y, float *device_y, float *device_x, float *device_k, const int B, const int M, const int C, const int H, const int W, const int K)
{	// Copy the output back to host
	const int hOut = H - K + 1;
    const int wOut = W - K + 1;
	cudaMemcpy(host_y, device_y, B*M*wOut*hOut*sizeof(float), cudaMemcpyDeviceToHost);//SIZE MAY BE WRONG
	
    // Free device memory
    cudaFree(device_y);
    cudaFree(device_x);
	
	// Useful snippet for error checking
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){	
	    std::cout<<"CUDA error Retrieval: "<<cudaGetErrorString(error)<<std::endl;
  	    exit(-1);
    }
}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}