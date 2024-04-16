
#ifndef __UTILS_H__
#define __UTILS_H__

#define FULLMASK 0xffffffffffffffff


#define SHFL_DOWN_MAX(v, temp_v)                     \
    temp_v = __shfl_down_sync(FULLMASK, v, 32);      \
    if (temp_v > v) {                                \
        v = temp_v;                                  \
    }                                                \
    temp_v = __shfl_down_sync(FULLMASK, v, 16);      \
    if (temp_v > v) {                                \
        v = temp_v;                                  \
    }                                                \
    temp_v = __shfl_down_sync(FULLMASK, v, 8);       \
    if (temp_v > v) {                                \
        v = temp_v;                                  \
    }                                                \
    temp_v = __shfl_down_sync(FULLMASK, v, 4);       \
    if (temp_v > v) {                                \
        v = temp_v;                                  \
    }                                                \
    temp_v = __shfl_down_sync(FULLMASK, v, 2);       \
    if (temp_v > v) {                                \
        v = temp_v;                                  \
    }                                                \
    temp_v = __shfl_down_sync(FULLMASK, v, 1);       \
    if (temp_v > v) {                                \
        v = temp_v;                                  \
    };

#define SHFL_DOWN_SUM(v)                     \
    v += __shfl_down_sync(FULLMASK, v, 32);  \
    v += __shfl_down_sync(FULLMASK, v, 16);  \
    v += __shfl_down_sync(FULLMASK, v, 8);   \
    v += __shfl_down_sync(FULLMASK, v, 4);   \
    v += __shfl_down_sync(FULLMASK, v, 2);   \
    v += __shfl_down_sync(FULLMASK, v, 1);


#endif


