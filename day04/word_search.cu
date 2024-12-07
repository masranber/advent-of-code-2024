
/*
CUDA doesn't have string functions from C stdlib so we will steal a simple
C implementation from https://stackoverflow.com/questions/32560167/strncmp-implementation

This code was written for regular C so there are
probably more efficient ways to do string compares on the GPU.
*/
__device__ int cuda_strncmp( const char *s1, const char *s2, unsigned int n) {
    while(n && *s1 && (*s1 == *s2)) {
        s1++;
        s2++;
        n--;
    }
    if(n == 0) return 0;
    return (*(unsigned char*)s1 - *(unsigned char*)s2);
}

/*
Must be run with 1024 threads per block and 1 block per grid.
Because we're limited to 1024 by CUDA, the max size of word search
this will work for is 511x511 I believe (diagonal has 2 extra rows)
*/
__global__ void word_search(char *word_search, int rows, int cols, int *result) {
    unsigned int tid = threadIdx.x;
    unsigned int row = tid % 512; // every row gets hit by 2 threads, one forward and one reverse
    unsigned int dir = tid / 512;

    if(row >= rows) return;

    char *word = dir ? "SAMX" : "XMAS"; // ideally the word gets passed in as a param

    for(int col = 0; col < (cols - 3); col++) { // -3 is for a word of length 4
        int i = (row * cols) + col;
        int not_eq = cuda_strncmp(&word_search[i], word, 4);
        atomicAdd(result, !not_eq); // strncmp returns 0 if eq
    }
}


/*

...S.
.XA..
.M.S.
XA.A.
.SAMX
...X.

.S.S.
.A...
XMAS.
.X...
.....
.....
.....
.....
XX...
.M...
.AA..

.X.AX
.AS..
.....
S....
.....
.....
.....
.....
XS...
.AAX.
.M.M.

*/

/*

0,5 *
1,5 -
4,0 *
4,7
9,5 *

*/
