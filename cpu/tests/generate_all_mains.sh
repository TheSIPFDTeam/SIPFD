#!/bin/bash

PRIMES="69 72 76 82 87 89 95 99 100 107 111 117 118 124 126 129 131 140 152 164 176 184 194 196 204 216 224 236 244 252"

for p in $PRIMES; do
    ./generate_mains.sh $p;
done;