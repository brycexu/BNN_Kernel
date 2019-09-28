#define ENCODE_BIT 32

void encode_rows_cpu_kernel(uint32_t *columns, uint32_t *columns_binary, int m, int n);

void encode_cols_cpu_kernel(uint32_t *columns, uint32_t *columns_binary, int m, int n);