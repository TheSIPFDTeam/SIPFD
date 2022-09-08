#if defined(_shortw_)
int main(){
    printf("Not implemented; use MODEL=mont\n");
    return -1;
}
#elif defined(_mont_)
#include "mitm-dfs-memory-limited_mont.c"
#endif
