#ifndef TVM_RELAY_PASSES_ADAPTIVE_FUSION_PASS_H_
#define TVM_RELAY_PASSES_ADAPTIVE_FUSION_PASS_H_

#include <tvm/relay/expr.h>
#include <tvm/relay/function.h>
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/attrs/transform.h>
#include <tvm/relay/type.h>
#include <tvm/relay/op_attr_types.h>

#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>

namespace tvm {
namespace relay {

/*!
 * \brief Record of a potential fusion edge between two operators
 */
struct EdgeRecord {
  /*! \brief Unique ID for the graph this edge belongs to */
  std::string graph_id;
  
  /*! \brief Unique ID for this edge */
  int edge_id;
  
  /*! \brief Operator ID (symbol) for first node */
  std::string op_a_id;
  
  /*! \brief Operator ID (symbol) for second node */
  std::string op_b_id;
  
  /*! \brief Hash of the shape of operator A's output */
  size_t a_shape_hash;
  
  /*! \brief Hash of the shape of operator B's input */
  size_t b_shape_hash;
  
  /*! \brief Flops estimate for operator A */
  int64_t flops_a;
  
  /*! \brief Flops estimate for operator B */
  int64_t flops_b;
  
  /*! \brief Data type of operator A's output */
  std::string dtype_a;
  
  /*! \brief Data type of operator B's input */
  std::string dtype_b;
  
  /*! \brief HBM memory free percentage at profiling time */
  float hbm_free;
  
  /*! \brief SM occupancy at profiling time */
  float sm_occ;
  
  /*! \brief Chiplet ID if applicable */
  int chiplet_id;
  
  /*! \brief Latency difference when this edge is fused vs not fused (ms) */
  float delta_latency_ms;
};

/*!
 * \brief Compute a hash for a tensor shape
 * \param type The tensor type with shape information
 * \return A hash value representing the shape
 */
size_t HashShape(const tvm::relay::Type& type);

/*!
 * \brief Serialize an EdgeRecord to a binary file
 * \param record The edge record to serialize
 * \param filepath The path to write the record to
 */
void SerializeToFile(const EdgeRecord& record, const std::string& filepath);

/*!
 * \brief Forced fusion pass for benchmarking a specific edge
 * \param edge_id The ID of the edge to force fusion for
 * \return The modified module with the specified edge fused
 */
tvm::relay::Function ForceFuseEdge(
  const tvm::relay::Function& func,
  int edge_id,
  const std::string& graph_id);

/*!
 * \brief Create the adaptive fusion pass that records edge candidates
 * \param dump_edges Whether to dump fusion edge records
 * \return The pass 
 */
tvm::transform::Pass CreateAdaptiveFusionPass(bool dump_edges = false);

}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_PASSES_ADAPTIVE_FUSION_PASS_H_ 