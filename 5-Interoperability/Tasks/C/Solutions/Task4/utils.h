#pragma once

#include <stdlib.h>
#include <sys/time.h>

#define CUFFTC(f) if(CUFFT_SUCCESS != f) { printf("cufft ERROR in line %d", __LINE__); exit(42); }

double getTime(void) {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + tv.tv_usec*1e-6;
}
