<rng1>
    else if (nbits <= 160) {
        a[0] = r[0].x;
        a[1] = r[0].y;
        a[2] = r[0].z;
        a[3] = r[0].w;
        a[4] = r[1].x & mask;
    } else if (nbits <= 192) {
        a[0] = r[0].x;
        a[1] = r[0].y;
        a[2] = r[0].z;
        a[3] = r[0].w;
        a[4] = r[1].x;
        a[5] = r[1].y & mask;
    }
