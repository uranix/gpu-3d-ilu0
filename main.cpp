// vim: noet ts=4 sts=4 sw=4

#include "block.h"
#include <iostream>

void dump(const SkewedBlock& block) {
	int ni = block.ni;
	int nj = block.nj;
	int nk = block.nk;
	for (int i = -1; i <= ni; i++) {
		std::cout << "i = " << i << std::endl;
		for (int j = -1; j <= nj; j++) {
			for (int k = -1; k <= nk; k++) {
				printf("%6.3f ", block(i, j, k));
			}
			std::cout << std::endl;
		}
	}

}

int main() {
	int ni = 5, nj = 4, nk = 3;
	SkewedBlock block(ni, nj, nk);
	SkewedBlock lap(ni, nj, nk);
	double hx = 0.1, hy = 0.2, hz = 0.15;

	for (double &v : block.data) {
		v = std::nan("");
	}

	for (int i = -1; i <= ni; i++) {
		bool xside = i == -1 || i == ni;
		for (int j = -1; j <= nj; j++) {
			bool yside = j == -1 || j == nj;
			for (int k = -1; k <= nk; k++) {
				bool zside = k == -1 || k == nk;
				if ((int)xside + (int)yside + (int)zside < 1) {
					block(i, j, k) = 0.1 * (i + 0.1 * j + 0.01 * k);
				}
			}
		}
	}

	for (int i = 0; i < ni; i++) {
		for (int j = 0; j < nj; j++) {
			for (int k = 0; k < nk; k++) {
				lap(i, j, k) = 
					block(i - 1, j, k) + block(i + 1, j, k) +
					block(i, j - 1, k) + block(i, j + 1, k) +
					block(i, j, k - 1) + block(i, j, k + 1) -
					6 * block(i, j, k);
			}
		}
	}

	std::vector<double> Sxm(nj * nk);
	std::vector<double> Sxp(nj * nk);
	std::vector<double> Sym(ni * nk);
	std::vector<double> Syp(ni * nk);
	std::vector<double> Szm(ni * nj);
	std::vector<double> Szp(ni * nj);

	std::cout << block.base << " " << block.di << " " << block.dj << " " << block.dk << std::endl;

	dump(block);

	extractSlow(block, Sxm.data(), Sxp.data(), Sym.data(), Syp.data(), Szm.data(), Szp.data());
	implodeSlow(block, Sxm.data(), Sxp.data(), Sym.data(), Syp.data(), Szm.data(), Szp.data());

	dump(block);

	return 0;
}
