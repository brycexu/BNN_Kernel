#include <TH/TH.h>
#include <stdio.h>
#include <stdint.h>
#include "binop_cpu_comp_kernel.h"

inline uint32_t encode_val(float* array, int n) {
    uint32_t sign, r = 0;
    for(int i=0; i<ENCODE_BIT && i<n; i++){
        sign = array[i]>0;
        r |= (sign<<i);
    }
    return r;
}

void encode_rows_cpu_kernel(float *columns, uint32_t *columns_binary, int m, int n) {
    int i, l = 1+(n-1)/ENCODE_BIT;
    #pragma omp parallel for
    for (i = 0; i < m*l; i++) {
        int p = n*(i/l)+ENCODE_BIT*(i%l);

        columns_binary[i] = encode_val(&columns[p], n-ENCODE_BIT*(i%l));
    }
}

void encode_cols_cpu_kernel(float *columns, uint32_t *columns_binary, int m, int n) {
    int col_bin_m = 1 + (m-1) / ENCODE_BIT;
    int i, j, k;
    #pragma omp parallel for
    for (i = 0; i < col_bin_m; i++) {
        int i64 = i * ENCODE_BIT;
        for (j = 0; j < n && i64<m ; j++) {

            uint32_t sign, rvalue = 0;

            for (k = 0; j + n * (i64 + k) < m*n && k < ENCODE_BIT; k++) {
                sign = columns[j + n * (i64 + k)]>0;
                rvalue |= (sign << k);
            }

            columns_binary[j + n * i] = rvalue;
        }
    }
}