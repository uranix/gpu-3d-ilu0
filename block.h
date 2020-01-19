// vim: noet ts=4 sts=4 sw=4
#include <vector>
#include <cmath>
#include <cassert>

struct biptr {
	double *host;
	double *dev;
	const size_t bytes;

	biptr(size_t elems);
	~biptr();

	void dtoh();
	void htod();
}

struct SkewedBlock {
	const int ni, nj, nk;
	const int nis, njs, nks;

	biptr data;
	biptr Sxm;
	biptr Sxp;
	biptr Sym;
	biptr Syp;
	biptr Szm;
	biptr Szp;

	int zerooffs, di, dj, dk;

	static int align(int n) {
		return (n + 15) & ~15;
	}

	SkewedBlock(int ni, int nj, int nk)
		: ni(ni), nj(nj), nk(nk)
		, nis(ni + 2), njs(nj + 2), nks(align(nk + 2))
		, data((ni + nj + nk + 4) * (nj + 2) * align(nk + 2))
	{
		double *base = &(*this)(0, 0, 0);
		zerooffs = base - data.host;
		di = &(*this)(1, 0, 0) - base;
		dj = &(*this)(0, 1, 0) - base;
		dk = &(*this)(0, 0, 1) - base;
	}

	void extract();
	void implode();

	double &operator()(int i, int j, int k) {
		assert(i >= -1 && i <= ni);
		assert(j >= -1 && j <= nj);
		assert(k >= -1 && k <= nk);
		return (*this)(i + j + k + 3, (j + 1) * nks + (k + 1));
	}
	const double &operator()(int i, int j, int k) const {
		assert(i >= -1 && i <= ni);
		assert(j >= -1 && j <= nj);
		assert(k >= -1 && k <= nk);
		return (*this)(i + j + k + 3, (j + 1) * nks + (k + 1));
	}
	double &operator()(int a, int b) {
		assert(a >= 0 && a <= ni + nj + nk + 3);
		assert(b >= 0 && b < njs * nks);
		return data.host[a * njs * nks + b];
	}
	const double &operator()(int a, int b) const {
		assert(a >= 0 && a <= ni + nj + nk + 3);
		assert(b >= 0 && b < njs * nks);
		return data.host[a * njs * nks + b];
	}
};

void extractSlow(const SkewedBlock& block,
			double *Sxm, double *Sxp,
			double *Sym, double *Syp,
			double *Szm, double *Szp);

void implodeSlow(SkewedBlock& block,
			const double *Sxm, const double *Sxp,
			const double *Sym, const double *Syp,
			const double *Szm, const double *Szp);

