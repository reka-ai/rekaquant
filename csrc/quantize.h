#include <stdint.h>


typedef uint16_t ggml_half;
typedef uint32_t ggml_half2;
typedef uint16_t ggml_fp16_t;


#define QK_K 256
#define K_SCALE_SIZE 12

typedef struct {
    union {
        struct {
            ggml_half d;    // super-block scale for quantized scales
            ggml_half dmin; // super-block scale for quantized mins
        };
        ggml_half2 dm;
    };
    uint8_t scales[K_SCALE_SIZE]; // scales and mins, quantized with 6 bits
    uint8_t qs[QK_K/2];           // 4--bit quants
} block_q4_K;


typedef struct {
    uint8_t hmask[QK_K/8]; // quants - high bit
    uint8_t qs[QK_K/4];    // quants - low 2 bits
    uint8_t scales[12];    // scales, quantized with 6 bits
    ggml_half d;           // super-block scale
} block_q3_K;

typedef struct {
    uint8_t ql[QK_K/2];      // quants, lower 4 bits
    uint8_t qh[QK_K/4];      // quants, upper 2 bits
    int8_t  scales[QK_K/16]; // scales, quantized with 8 bits
    ggml_half d;             // super-block scale
} block_q6_K;


void quantize_row_q4_K_ref(const float *  x, block_q4_K *  y, int64_t k);
void dequantize_row_q4_K(const block_q4_K *  x, float *  y, int64_t k);

void quantize_row_q3_K_ref(const float *  x, block_q3_K *  y, int64_t k);
void dequantize_row_q3_K(const block_q3_K *  x, float *  y, int64_t k);

void quantize_row_q6_K_ref(const float *  x, block_q6_K *  y, int64_t k);
void dequantize_row_q6_K(const block_q6_K *  x, float *  y, int64_t k);