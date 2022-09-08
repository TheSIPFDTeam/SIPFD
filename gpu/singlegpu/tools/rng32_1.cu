    if (nbits <= 32) {
        a[0] = r[0].x & mask;
    } else if (nbits <= 64) {
        a[0] = r[0].x;
        a[1] = r[0].y & mask;
    } else if(nbits <= 96) {
        a[0] = r[0].x;
        a[1] = r[0].y;
        a[2] = r[0].z & mask;
    } else if (nbits <= 128) {
        a[0] = r[0].x;
        a[1] = r[0].y;
        a[2] = r[0].z;
        a[3] = r[0].w & mask;
    }
