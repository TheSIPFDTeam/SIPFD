    if (nbits <= 32) {
        a[0] = (uint64_t)r[0].x & mask;
    } else if (nbits <= 64) {
        a[0] = (uint64_t)r[0].x;
        a[0] ^= ((uint64_t)r[0].y & mask) << 32;
    } else if(nbits <= 96) {
        a[0] = (uint64_t)r[0].x;
        a[0] ^= ((uint64_t)r[0].y) << 32;
        a[1] = r[0].z & mask;
    } else if (nbits <= 128) {
        a[0] = (uint64_t)r[0].x;
        a[0] ^= ((uint64_t)r[0].y) << 32;
        a[1] = (uint64_t)r[0].z;
        a[1] ^= ((uint64_t)r[0].w & mask) << 32;
    }
