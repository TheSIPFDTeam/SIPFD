<rng2>
    else {
        a[0] = (uint64_t)r[0].x;
        a[0] ^= ((uint64_t)r[0].y) << 32;
        a[1] = (uint64_t)r[0].z;
        a[1] ^= (uint64_t)r[0].w << 32;
        a[2] = (uint64_t)r[1].x;
        a[2] ^= (uint64_t)r[1].y << 32;
        a[3] = (uint64_t)r[1].z;
        a[3] ^= ((uint64_t)r[1].w & mask) << 32;
    }
