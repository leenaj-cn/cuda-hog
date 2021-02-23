#pragma once
#include <stdio.h>

#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include <opencv2/opencv.hpp>
using namespace cv;

#define HISTOGRAM64_BIN_COUNT 64
#define HISTOGRAM256_BIN_COUNT 256
#define UINT_BITS 32

typedef unsigned int uint;
typedef unsigned char uchar;

