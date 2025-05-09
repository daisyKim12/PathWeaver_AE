#include <pybind11/pybind11.h>
#include <pybind11/numpy.h> 
#include <cstdint>
#include <vector>

#include "common.cuh"
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/neighbors/cagra.cuh>
#include <raft/neighbors/cagra_types.hpp>
#include <raft/random/make_blobs.cuh>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <cstdint>
#include <vector>

namespace py = pybind11;

class CagraWrapper {
public:
    CagraWrapper() : dev_resources() {
        rmm::mr::set_current_device_resource(&pool_mr);
    }

    void run_build_search(py::array_t<float> dataset, py::array_t<float> queries, int64_t topk);
    py::array_t<uint32_t> build(py::array_t<float> dataset);
    void search(py::array_t<float> dataset, py::array_t<uint32_t> graph, py::array_t<float> queries, int64_t topk);
    py::array_t<uint32_t> get_neighbors();
    py::array_t<float> get_distances();

private:
    raft::device_resources dev_resources;

    // Set pool memory resource with 1 GiB initial pool size. All allocations use the same pool.
    rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource> pool_mr{
        rmm::mr::get_current_device_resource(), 1024 * 1024 * 1024ull};

    py::array_t<uint32_t> result_neighbors;
    py::array_t<float> result_distances;

};

void CagraWrapper::run_build_search(py::array_t<float> dataset, py::array_t<float> queries, int64_t topk) {
    int64_t n_raw_data = dataset.shape(0);
    int64_t n_dim = dataset.shape(1);
    int64_t n_queries = queries.shape(0);

    auto d_dataset = raft::make_device_matrix<float, int64_t>(dev_resources, n_raw_data, n_dim);
    auto d_queries = raft::make_device_matrix<float, int64_t>(dev_resources, n_queries, n_dim);

    // Copy data to device
    raft::update_device(d_dataset.data_handle(), dataset.data(), n_raw_data * n_dim, dev_resources.get_stream());
    raft::update_device(d_queries.data_handle(), queries.data(), n_queries * n_dim, dev_resources.get_stream());

    // Build Cagra graph (knn + optimize)
    raft::neighbors::cagra::index_params index_params;
    std::cout << "Building CAGRA index" << std::endl;
    auto index = raft::neighbors::cagra::build<float, uint32_t>(dev_resources, index_params, raft::make_const_mdspan(d_dataset.view()));
    std::cout << "CAGRA index has " << index.size() << " vectors" << std::endl;
    std::cout << "CAGRA graph has degree " << index.graph_degree() << ", graph size [" << index.graph().extent(0) << ", " << index.graph().extent(1) << "]" << std::endl;

    // Search on Cagra graph
    auto d_neighbors = raft::make_device_matrix<uint32_t>(dev_resources, n_queries, topk);
    auto d_distances = raft::make_device_matrix<float>(dev_resources, n_queries, topk);
    std::cout << "Search on built CAGRA graph index" << std::endl;
    raft::neighbors::cagra::search_params search_params;
    raft::neighbors::cagra::search<float, uint32_t>(dev_resources, search_params, index, raft::make_const_mdspan(d_queries.view()), d_neighbors.view(), d_distances.view());

    // Copy results back to host
    std::vector<uint32_t> h_neighbors(n_queries * topk);
    std::vector<float> h_distances(n_queries * topk);
    raft::update_host(h_neighbors.data(), d_neighbors.data_handle(), n_queries * topk, dev_resources.get_stream());
    raft::update_host(h_distances.data(), d_distances.data_handle(), n_queries * topk, dev_resources.get_stream());

    // Print results
    //print_results(dev_resources, d_neighbors.view(), d_distances.view());

    // Store result
    result_neighbors = py::array_t<uint32_t>({n_queries, topk}, h_neighbors.data());
    result_distances = py::array_t<float>({n_queries, topk}, h_distances.data());

}

py::array_t<uint32_t> CagraWrapper::build(py::array_t<float> dataset) {
    int64_t n_raw_data = dataset.shape(0);
    int64_t n_dim = dataset.shape(1);

    auto d_dataset = raft::make_device_matrix<float, int64_t>(dev_resources, n_raw_data, n_dim);
    // Copy data to device
    raft::update_device(d_dataset.data_handle(), dataset.data(), n_raw_data * n_dim, dev_resources.get_stream());

    // Build Cagra graph (knn + optimize)
    raft::neighbors::cagra::index_params index_params;
    // auto index = raft::neighbors::cagra::build<float, uint32_t>(dev_resources, index_params, raft::make_const_mdspan(d_dataset.view()));
    raft::neighbors::cagra::index<float, uint32_t> index = raft::neighbors::cagra::build<float, uint32_t>(dev_resources, index_params, raft::make_const_mdspan(d_dataset.view()));
    std::cout << "Start CAGRA - Building with #Node = "<< index.graph().extent(0) << ", k = " << index.graph().extent(1) << std::endl;

    // Copy graph data back to host and create py::array_t
    int64_t graph_rows = index.graph().extent(0);
    int64_t graph_cols = index.graph().extent(1);
    std::vector<uint32_t> h_graph(graph_rows * graph_cols);

    raft::update_host(h_graph.data(), index.graph().data_handle(), graph_rows * graph_cols, dev_resources.get_stream());

    // Return the graph as py::array_t
    return py::array_t<uint32_t>({graph_rows, graph_cols}, h_graph.data());

}

void CagraWrapper::search(py::array_t<float> dataset, py::array_t<uint32_t> graph, py::array_t<float> queries, int64_t topk) {

    int64_t n_raw_data = dataset.shape(0);
    int64_t n_dim = dataset.shape(1);
    int64_t n_queries = queries.shape(0);


    auto d_dataset = raft::make_device_matrix<float, int64_t>(dev_resources, n_raw_data, n_dim);    
    auto d_queries = raft::make_device_matrix<float, int64_t>(dev_resources, n_queries, n_dim);

    // Copy data to device
    raft::update_device(d_dataset.data_handle(), dataset.data(), n_raw_data * n_dim, dev_resources.get_stream());
    raft::update_device(d_queries.data_handle(), queries.data(), n_queries * n_dim, dev_resources.get_stream());

    // Reconstruct cagra graph index
    raft::neighbors::cagra::index_params _params;
    //mdspan<const T, matrix_extent<int64_t>, row_major, Accessor> _dataset = raft::make_const_mdspan(d_dataset.view());
    auto _dataset = raft::make_const_mdspan(d_dataset.view());
   
    //auto cagra_graph = raft::make_host_matrix<float, int64_t>(graph.shape[0], graph.shape[1]);
    // Copy graph to device
    py::buffer_info graph_buf = graph.request();
    auto graph_ptr = static_cast<uint32_t*>(graph_buf.ptr);
    auto d_cagra_graph = raft::make_device_matrix<uint32_t, int64_t>(dev_resources, graph.shape(0), graph.shape(1));
    raft::update_device(d_cagra_graph.data_handle(), graph_ptr, graph.shape(0) * graph.shape(1), dev_resources.get_stream());

    
    raft::neighbors::cagra::index<float, uint32_t> cagra_index(dev_resources, _params.metric, _dataset, raft::make_const_mdspan(d_cagra_graph.view()));


    // Search on Cagra graph
    auto d_neighbors = raft::make_device_matrix<uint32_t>(dev_resources, n_queries, topk);
    auto d_distances = raft::make_device_matrix<float>(dev_resources, n_queries, topk);
    std::cout << "Search on built CAGRA graph index" << std::endl;
    raft::neighbors::cagra::search_params search_params;
    raft::neighbors::cagra::search<float, uint32_t>(dev_resources, search_params, cagra_index, raft::make_const_mdspan(d_queries.view()), d_neighbors.view(), d_distances.view());

    // Copy results back to host
    std::vector<uint32_t> h_neighbors(n_queries * topk);
    std::vector<float> h_distances(n_queries * topk);
    raft::update_host(h_neighbors.data(), d_neighbors.data_handle(), n_queries * topk, dev_resources.get_stream());
    raft::update_host(h_distances.data(), d_distances.data_handle(), n_queries * topk, dev_resources.get_stream());

    // Print results
    // print_results(dev_resources, d_neighbors.view(), d_distances.view());

    // Store result
    result_neighbors = py::array_t<uint32_t>({n_queries, topk}, h_neighbors.data());
    result_distances = py::array_t<float>({n_queries, topk}, h_distances.data());

}


py::array_t<uint32_t> CagraWrapper::get_neighbors(){
    return result_neighbors;
}

py::array_t<float> CagraWrapper::get_distances(){
    return result_distances;
}


PYBIND11_MODULE(cagra_wrapper, m) {                                         // cagra는 모듈 이름이고 CMakeLists에 있는 모듈 이름과 동일해야 한다. m은 모듈 객체이다.
    
    m.doc() = "pybind11 cagra plugin";
    
    py::class_<CagraWrapper>(m, "CagraWrapper")
        .def(py::init<>())
        .def("run_build_search", &CagraWrapper::run_build_search, "Build Cagra graph and search", py::arg("dataset"), py::arg("queries"), py::arg("topk"))
        .def("build", &CagraWrapper::build, "Build Cagra graph", py::arg("dataset"))
        .def("search", &CagraWrapper::search, "Search on Cagra graph", py::arg("dataset"), py::arg("graph"), py::arg("queried"), py::arg("topk"))
        .def("get_neighbors", &CagraWrapper::get_neighbors, "Get the neighbors results")
        .def("get_distances", &CagraWrapper::get_distances, "Get the distances results");
    
}
