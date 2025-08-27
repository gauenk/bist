


std::tuple<uint32_t*, uint32_t*>
get_csr_graph_from_edges(uint32_t* edges, uint8_t* ebids, int* eptr, int* vptr, int V_total, int E_total);

std::tuple<uint32_t*, int*>
get_edges_from_csr(uint32_t* csr_edges, uint32_t* csr_eptr, int* vptr, uint8_t* vbids, int V, int B);