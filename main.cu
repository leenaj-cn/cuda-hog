#include "Header.cuh"



__global__ void gray_cu(uchar* src, uchar* dst, uint w, uint h)
{
	uint col = blockDim.x * blockIdx.x + threadIdx.x;
	uint row = blockDim.y * blockIdx.y + threadIdx.y;
	
	uint srcStep = w * 3;
	uint dstStep = w;

	if (row < h & col < w)
	{
		//BGR
		dst[row * dstStep + col] = 0.114 * src[row * srcStep + col*3] + 0.587 * src[row * srcStep + col * 3 + 1] + 0.299 * src[row * srcStep + col * 3 + 2];
	}
}

__global__ void hist_256_cu(uchar* data, uint* hist_256, uint width, uint height)
{
	uint col = blockDim.x * blockIdx.x + threadIdx.x;
	uint row = blockDim.y * blockIdx.y + threadIdx.y;
	uint pos = row * width + col;

	if (col < width & row < height)
	{
		int value = data[pos];
		atomicAdd(&hist_256[value], 1);
		__syncthreads();
	}


}


void gray_kerbel(uchar* d_srcData, uchar* d_dstData, uint width, uint height)
{

	dim3 block(16, 16);
	dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	cudaEventQuery(start);


	gray_cu << <grid, block >> > (d_srcData, d_dstData, width, height);

	cudaGetLastError();
	cudaDeviceSynchronize();


	//time end
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime_cuda;
	cudaEventElapsedTime(&elapsedTime_cuda, start, stop);
	printf("rgb 2 gray kernel time=%f ms\n", elapsedTime_cuda);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
}


void hist_256_kernel(uchar* d_grayData, uint* d_hist_256, uint width, uint height)
{
	dim3 block(16, 16);
	dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	cudaEventQuery(start);

	hist_256_cu << <grid, block >> > (d_grayData, d_hist_256, width, height);

	cudaGetLastError();
	cudaDeviceSynchronize();

	//time end
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime_cuda;
	cudaEventElapsedTime(&elapsedTime_cuda, start, stop);
	printf("hist_256_cu kernel time=%f ms\n", elapsedTime_cuda);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

}

void histogramCUDA(Mat src, Mat gray, int* hist_256)
{
	uint width = src.cols;
	uint height = src.rows;
	uint imagesize = width * height * sizeof(uchar);

	uchar* d_srcData = gray.data;
	uchar* d_grayData;

	//rgb2gray image
	cudaMalloc((void**)&d_srcData, imagesize * 3);
	cudaMalloc((void**)&d_grayData, imagesize);
	cudaMemcpy(d_srcData, src.data, imagesize * 3, cudaMemcpyHostToDevice);

	//histogram 256 bin
	uint* d_hist_256;
	uint hist_256_size = 256 * sizeof(uint);
	cudaMalloc((void**)&d_hist_256, hist_256_size);

	//kernel
	gray_kerbel(d_srcData, d_grayData, width, height);
	hist_256_kernel(d_grayData, d_hist_256, width, height);


	cudaMemcpy(gray.data, d_grayData, imagesize, cudaMemcpyDeviceToHost);
	cudaMemcpy(hist_256, d_hist_256, hist_256_size, cudaMemcpyDeviceToHost);

	cudaFree(d_srcData);
	cudaFree(d_grayData);

}


int main()
{
	Mat srcImage;
	srcImage = imread("cat.jpg");
	if (srcImage.empty()) {
		printf("Error! Input image failed to be load!\n");
		return -1;
	}

	Mat grayImage(srcImage.rows, srcImage.cols, CV_8UC1);
	int hist_256[HISTOGRAM256_BIN_COUNT] = { 0 };
	
	histogramCUDA(srcImage, grayImage, hist_256);
	//cvtColor(srcImage, grayImage, COLOR_BGR2GRAY);

	uint width = grayImage.cols;
	uint height = grayImage.rows;
	uint imgagesize = width * height * grayImage.channels();
	uchar* imagedata = grayImage.data;






	//check result
	uint hist_256_golden[HISTOGRAM256_BIN_COUNT] = { 0 };
	for (int i = 0; i < imgagesize; i++)
	{
		int c = imagedata[i];
		hist_256_golden[c]++;
	}

	for (int j = 0; j < HISTOGRAM256_BIN_COUNT; j++)
	{
		
		printf("hist_256[%d]:%d, hist_256_golden[%d]:%d\n", j, hist_256[j],j, hist_256_golden[j]);
	}
	



	//show image in the window
	namedWindow("src.jpg", WINDOW_AUTOSIZE);
	namedWindow("gray.jpg", WINDOW_AUTOSIZE);

	imshow("src.jpg", srcImage);
	imshow("gray.jpg", grayImage);

	imwrite("gray.png", grayImage);

	printf("Press any key to exit...\n");

	return 0;
}




