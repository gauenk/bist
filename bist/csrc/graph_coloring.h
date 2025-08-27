

uint8_t* get_graph_coloring(
    const uint32_t* edges,  // CSR edge list
    const uint32_t* eptr,       // CSR edge pointers (size V+1)
    int num_vertices);