
__global__ void antenna_antinodes(char *antenna_map, int *antinode_map, int rows, int cols) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;

    if(i >= total) return;

    char ant1 = antenna_map[i];
    if(ant1 == '.') return;

    // each thread is assigned a grid position (may or may not be an antenna)
    // and is responsible for
    // determining its antinodes with all other same-freq antennas
    // essentially turning an O(n^2) problem into an O(n) problem
    // in reality it's not really O(n) when n > # GPU threads but CUDA provides
    // the illusion it's O(n) to the kernel (one loop).
    // so it's not perfect but it's still taking advantage of parallel compute
    for(int j = ((i + 1) % total); j != i; j = ((j + 1) % total)) {
        char ant2 = antenna_map[j];
        if(ant1 == ant2) {
            int i_x = i % cols;
            int i_y = i / cols;
            int delta_x = (j % cols) - (i % cols);
            int delta_y = (j / cols) - (i / cols);
            int antinode_x = i_x + (delta_x * 2);
            int antinode_y = i_y + (delta_y * 2);
            if(antinode_y >= 0 && antinode_y < rows && antinode_x >= 0 && antinode_x < cols) {
                int antinode_i = antinode_y * cols + antinode_x;
                antinode_map[antinode_i] = 1;
            }
        }
    }
}

__global__ void antenna_antinodes_harmonic(char *antenna_map, int *antinode_map, int rows, int cols) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;

    if(i >= total) return;

    char ant1 = antenna_map[i];
    if(ant1 == '.') return;

    // each thread is assigned a grid position (may or may not be an antenna)
    // and is responsible for
    // determining its antinodes with all other same-freq antennas
    // essentially turning an O(n^2) problem into an O(n) problem
    // in reality it's not really O(n) when n > # GPU threads but CUDA provides
    // the illusion it's O(n) to the kernel (one loop).
    // so it's not perfect but it's still taking advantage of parallel compute
    for(int j = ((i + 1) % total); j != i; j = ((j + 1) % total)) {
        char ant2 = antenna_map[j];
        if(ant1 == ant2) {
            int i_x = i % cols;
            int i_y = i / cols;
            int delta_x = (j % cols) - (i % cols);
            int delta_y = (j / cols) - (i / cols);
            int antinode_x = i_x + delta_x; // new start pos is at antenna
            int antinode_y = i_y + delta_y;
            // step in direction of delta vector (   T-> T)
            while(antinode_y >= 0 && antinode_y < rows && antinode_x >= 0 && antinode_x < cols) {
                int antinode_i = antinode_y * cols + antinode_x;
                antinode_map[antinode_i] = 1;
                antinode_x += delta_x;
                antinode_y += delta_y;
            }
            // step in inverse direction of delta vector ( <-T   T)
            while(antinode_y >= 0 && antinode_y < rows && antinode_x >= 0 && antinode_x < cols) {
                int antinode_i = antinode_y * cols + antinode_x;
                antinode_map[antinode_i] = 1;
                antinode_x -= delta_x;
                antinode_y -= delta_y;
            }
        }
    }
}
