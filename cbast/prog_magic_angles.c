#include <math.h>
#include <stdlib.h>
#include <stdio.h>
int main(int argc, char **argv) {
	int n = atoi(argv[1]);
	for (int i = 0; i < n; i++) {
		float theta = 2 * asin(asin(1./(2*sqrt(3*i*i+3*i+1))));
		printf("%f\n", theta * 180 / M_PI);
	}
}
