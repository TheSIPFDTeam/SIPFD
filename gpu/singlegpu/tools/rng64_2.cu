<rng1>
    else if (nbits <= 160) {
        a[0] = (uint64_t)r[0].x;
        a[0] ^= ((uint64_t)r[0].y) << 32;
        a[1] = (uint64_t)r[0].z;
        a[1] ^= (uint64_t)r[0].w << 32;
        a[2] = (uint64_t)r[1].x & mask;
    } else if (nbits <= 192) {
        a[0] = (uint64_t)r[0].x;
        a[0] ^= ((uint64_t)r[0].y) << 32;
        a[1] = (uint64_t)r[0].z;
        a[1] ^= (uint64_t)r[0].w << 32;
        a[2] = (uint64_t)r[1].x;
        a[2] ^= ((uint64_t)r[1].y & mask) << 32;
    }
