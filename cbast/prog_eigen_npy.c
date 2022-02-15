#include <stdlib.h>
#include <stdio.h>
#include <lapacke.h>
#include <cblas.h>
#include "core.h"
#include "cnpy.h"

int main() {
	shaped_array_t Srr = NumpyLoadBinary("./Srr_disc.npy");
	const lapack_int N = Srr.shape[0];
	printf("%d\n", N);
	shaped_array_t Sri = NumpyLoadBinary("./Sri_disc.npy");
	shaped_array_t Slr = NumpyLoadBinary("./Slr_disc.npy");
	shaped_array_t Sli = NumpyLoadBinary("./Sli_disc.npy");
	printf("Loaded matrices\n");


	double * A = calloc(2*N*N, sizeof(double));
	double * B = calloc(2*N*N, sizeof(double));
	double * VL = calloc(2*N*N, sizeof(double));
	double * VR = calloc(2*N*N, sizeof(double));
	double * work = calloc(2*N*N, sizeof(double));
	double * alpha = calloc(2*N, sizeof(double));
	double * alpha_i = calloc(2*N, sizeof(double));
	double * beta = calloc(2*N, sizeof(double));

	for (int i = 0; i < N*N; i++)
	{
		A[2*i] = Srr.data[i];
		A[1+2*i] = Sri.data[i];
		B[2*i] = Slr.data[i];
		B[1+2*i] = Sli.data[i];
	}
	printf("Formatted matrices\n");

	lapack_int info = 0;

	info = LAPACKE_zggev3(CblasRowMajor, 'N', 'V', N, A, N, B, N,
                           alpha, beta,
                           VL, N,
                           VR, N);
	printf("Saving eigenspace\n");
	NumpyBinarySave(alpha, "alpha_d.npy", ARRSHAPE(N,2));
	NumpyBinarySave(beta, "beta_d.npy", ARRSHAPE(N,2));
	NumpyBinarySave(VL, "vr_d.npy", ARRSHAPE(N, N, 2));
	NumpyBinarySave(VR, "vl_d.npy", ARRSHAPE(N, N, 2));
	return 0;
}
