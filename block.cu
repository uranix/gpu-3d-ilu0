// vim: noet ts=4 sts=4 sw=4

#include "block.h"
#include <iostream>

void cudaCheckError(cudaError_t error, const char *file, const int line) {
	std::cerr << file << ":" << line << " CUDA error " << cudaGetErrorString(error) << std::endl;
	std::terminate();
}

#define CUDA_CHECK_ERROR(__err) cudaCheckError(x, __FILE__, __LINE__)

biptr::biptr(size_t elems) : bytes(elems * sizeof(double)) {
	cudaCheckError(cudaMallocHost(&host, bytes));
	cudaCheckError(cudaMalloc(&dev, bytes));
}

~biptr::biptr(size_t elems) {
	cudaCheckError(cudaFreeHost(host));
	cudaCheckError(cudaFree(dev));
}

void biptr::dtoh() {
	cudaCheckError(cudaMemcpy(host, device, bytes, cudaMemcpyDeviceToHost));
}

void biptr::htod() {
	cudaCheckError(cudaMemcpy(device, host, bytes, cudaMemcpyHostToDevice));
}

void extractSlow(const SkewedBlock& block,
			double *Sxm, double *Sxp,
			double *Sym, double *Syp,
			double *Szm, double *Szp)
{
	int ni = block.ni;
	int nj = block.nj;
	int nk = block.nk;
	for (int j = 0; j < nj; j++) {
		for (int k = 0; k < nk; k++) {
			Sxm[j * nk + k] = block(0   , j, k);
			Sxp[j * nk + k] = block(ni-1, j, k);
		}
	}
	for (int i = 0; i < ni; i++) {
		for (int k = 0; k < nk; k++) {
			Sym[i * nk + k] = block(i, 0   , k);
			Syp[i * nk + k] = block(i, nj-1, k);
		}
	}
	for (int i = 0; i < ni; i++) {
		for (int j = 0; j < nj; j++) {
			Szm[i * nj + j] = block(i, j, 0   );
			Szp[i * nj + j] = block(i, j, nk-1);
		}
	}
}

void implodeSlow(SkewedBlock& block,
			const double *Sxm, const double *Sxp,
			const double *Sym, const double *Syp,
			const double *Szm, const double *Szp)
{
	int ni = block.ni;
	int nj = block.nj;
	int nk = block.nk;
	for (int j = 0; j < nj; j++) {
		for (int k = 0; k < nk; k++) {
			block(-1, j, k) = Sxm[j * nk + k];
			block(ni, j, k) = Sxp[j * nk + k];
		}
	}
	for (int i = 0; i < ni; i++) {
		for (int k = 0; k < nk; k++) {
			block(i, -1, k) = Sym[i * nk + k];
			block(i, nj, k) = Syp[i * nk + k];
		}
	}
	for (int i = 0; i < ni; i++) {
		for (int j = 0; j < nj; j++) {
			block(i, j, -1) = Szm[i * nj + j];
			block(i, j, nk) = Szp[i * nj + j];
		}
	}
}

void SkewedBlock::extract() {
	extractKernel<<<6, std::max(nj, nk)>>>(
			data.dev, ni, nj, nk,
			di, dj, dk,
			Sxm.dev, Sxp.dev,
			Sym.dev, Syp.dev,
			Szm.dev, Szp.dev);
}

void SkewedBlock::implode() {
	implodeKernel<<<6, std::max(nj, nk)>>>(
			data.dev, ni, nj, nk,
			di, dj, dk,
			Sxm.dev, Sxp.dev,
			Sym.dev, Syp.dev,
			Szm.dev, Szp.dev);
}

__global__ void extractKernel(
		const double *dataBase,
		const int ni, const int nj, const int nk,
		const int di, const int dj, const int dk,
		double *Sxm, double *Sxp,
		double *Sym, double *Syp,
		double *Szm, double *Szp)
{
	int side = blockIdx.x;
	int i, j, k;
	double *dst;
	if (side == 0) { // x-
		i = 0;
		dst = Sxm;
	} else if (side == 1) { // x+
		i = ni-1;
		dst = Sxp;
	} else if (side == 2) { // y-
		j = 0;
		dst = Sym;
	} else if (side == 3) { // y+
		j = nj-1;
		dst = Syp;
	} else if (side == 4) { // z-
		k = 0;
		dst = Szm;
	} else if (side == 5) { // z+
		k = nk-1;
		dst = Szp;
	}
	if (side == 0 || side == 1) {
		k = threadIdx.x;
		double *ptr = dataBase + i * di + k * dk;
		if (k < nk) {
			for (j = 0; j < nj; j++) {
				dst[j * nk + k] = ptr[j * dj];
			}
		}
	} else if (side == 2 || side == 3) {
		k = threadIdx.x;
		double *ptr = dataBase + j * dj + k * dk;
		if (k < nk) {
			for (i = 0; i < ni; i++) {
				dst[i * nk + k] = ptr[i * di];
			}
		}
	} else {
		j = threadIdx.x;
		double *ptr = dataBase + j * dj + k * dk;
		if (j < nj) {
			for (i = 0; i < ni; i++) {
				dst[i * nj + j] = ptr[i * di];
			}
		}
	}
}

__global__ void implodeKernel(
		const double *dataBase,
		const int ni, const int nj, const int nk,
		const int di, const int dj, const int dk,
		double *Sxm, double *Sxp,
		double *Sym, double *Syp,
		double *Szm, double *Szp)
{
	int side = blockIdx.y;
	int i, j, k;
	double *src;
	if (side == 0) { // x-
		i = -1;
		src = Sxm;
	} else if (side == 1) { // x+
		i = ni;
		src = Sxp;
	} else if (side == 2) { // y-
		j = -1;
		src = Sym;
	} else if (side == 3) { // y+
		j = nj;
		src = Syp;
	} else if (side == 4) { // z-
		k = -1;
		src = Szm;
	} else if (side == 5) { // z+
		k = nk;
		src = Szp;
	}
	if (side == 0 || side == 1) {
		k = threadIdx.x;
		double *ptr = dataBase + i * di + k * dk;
		if (k < nk) {
			for (j = 0; j < nj; j++) {
				ptr[j * dj] = src[j * nk + k];
			}
		}
	} else if (side == 2 || side == 3) {
		k = threadIdx.x;
		double *ptr = dataBase + j * dj + k * dk;
		if (k < nk) {
			for (i = 0; i < ni; i++) {
				ptr[i * di] = src[i * nk + k];
			}
		}
	} else {
		j = threadIdx.x;
		double *ptr = dataBase + j * dj + k * dk;
		if (j < nj) {
			for (i = 0; i < ni; i++) {
				ptr[i * di] = src[i * nj + j];
			}
		}
	}
}
