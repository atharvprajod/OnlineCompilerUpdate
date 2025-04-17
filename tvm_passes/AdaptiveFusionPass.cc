#include "AdaptiveFusionPass.h"

#include <tvm/relay/analysis.h>
#include <tvm/relay/attrs/memory.h>
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/transform.h>
#include <tvm/runtime/registry.h>

#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <string>
#include <fstream>
#include <functional>
#include <sstream>

namespace tvm {
namespace relay {

// Global counter for edge IDs
static int running_edge_counter = 0;

// Hash function for tensor shapes
size_t HashShape(const tvm::relay::Type& type) {
  if (!type.defined()) {
    return 0;
  }
  
  if (auto* tensor_type = type.as<TensorTypeNode>()) {
    std::string shape_str;
    for (auto dim : tensor_type->shape) {
      shape_str += std::to_string(dim.as<IntImmNode>()->value) + "_";
    }
    shape_str += tensor_type->dtype.to_string();
    return std::hash<std::string>{}(shape_str);
  }
  
  return 0;
}

// Estimate FLOPs for an operator (simplified)
int64_t EstimateFlops(const CallNode* call) {
  if (!call) return 0;
  
  // This is a simplified implementation. A real one would handle different op types.
  if (auto* conv2d = call->attrs.as<Conv2DAttrs>()) {
    auto* input_type = call->args[0]->checked_type().as<TensorTypeNode>();
    auto* weight_type = call->args[1]->checked_type().as<TensorTypeNode>();
    
    int batch = input_type->shape[0].as<IntImmNode>()->value;
    int in_c = input_type->shape[1].as<IntImmNode>()->value;
    int h = input_type->shape[2].as<IntImmNode>()->value;
    int w = input_type->shape[3].as<IntImmNode>()->value;
    
    int out_c = weight_type->shape[0].as<IntImmNode>()->value;
    int k_h = weight_type->shape[2].as<IntImmNode>()->value;
    int k_w = weight_type->shape[3].as<IntImmNode>()->value;
    
    // Simplified FLOP calculation for Conv2D
    int64_t flops = 2LL * batch * out_c * h * w * in_c * k_h * k_w / 
                   (conv2d->strides[0] * conv2d->strides[1]);
    return flops;
  }
  
  // Default estimate for other ops (very simplified)
  return 1000;
}

// Extract the data type of a tensor
std::string GetDType(const Type& type) {
  if (auto* tensor_type = type.as<TensorTypeNode>()) {
    return tensor_type->dtype.to_string();
  }
  return "unknown";
}

// Serialize edge record to file
void SerializeToFile(const EdgeRecord& record, const std::string& filepath) {
  // Ensure directory exists
  std::string dir = filepath.substr(0, filepath.find_last_of('/'));
  std::string cmd = "mkdir -p " + dir;
  system(cmd.c_str());
  
  // For this implementation, we'll use CSV format for simplicity
  // In a real implementation, you might use a binary format or a database
  std::ofstream file(filepath, std::ios::app);
  if (!file.is_open()) {
    LOG(WARNING) << "Could not open file for writing edge record: " << filepath;
    return;
  }
  
  // Write header if file is empty
  file.seekp(0, std::ios::end);
  if (file.tellp() == 0) {
    file << "graph_id,edge_id,op_a,op_b,flops_a,flops_b,shape_hash_a,shape_hash_b,"
         << "dtype_a,dtype_b,hbm_free,sm_occ,chiplet_id,delta_latency_ms\n";
  }
  
  // Write the record
  file << record.graph_id << ","
       << record.edge_id << ","
       << record.op_a_id << ","
       << record.op_b_id << ","
       << record.flops_a << ","
       << record.flops_b << ","
       << record.a_shape_hash << ","
       << record.b_shape_hash << ","
       << record.dtype_a << ","
       << record.dtype_b << ","
       << record.hbm_free << ","
       << record.sm_occ << ","
       << record.chiplet_id << ","
       << record.delta_latency_ms << "\n";
  
  file.close();
}

// Modified FuseOps pass that records fusion candidates
class AdaptiveFusionMutator : public ExprMutator {
 public:
  AdaptiveFusionMutator(bool dump_edges, std::string graph_id)
      : dump_edges_(dump_edges), graph_id_(graph_id) {}
  
  Expr VisitExpr_(const CallNode* call) override {
    // First visit the call normally
    Expr expr = ExprMutator::VisitExpr_(call);
    
    // Now check if we can fuse this call with any of its input calls
    for (size_t i = 0; i < call->args.size(); ++i) {
      if (auto* arg_call = call->args[i].as<CallNode>()) {
        // Check if the fusion is legal according to TVM's rules
        if (IsLegalPair(arg_call, call)) {
          // If we're dumping edges, record this candidate
          if (dump_edges_) {
            RecordFusionCandidate(arg_call, call);
          }
          
          // Here we would actually perform the fusion
          // In the real implementation, this would call MergeComposite or similar
          // For this skeleton, we just return the original expression
        }
      }
    }
    
    return expr;
  }
  
 private:
  bool dump_edges_;
  std::string graph_id_;
  
  // Simplified check for legal fusion pairs 
  bool IsLegalPair(const CallNode* a, const CallNode* b) {
    // In a real implementation, this would:
    // 1. Check data dependencies
    // 2. Check operator compatibility
    // 3. Check pattern support
    // 4. Apply target-specific fusion rules
    
    // For this skeleton, we'll assume simple heuristics
    if (!a || !b) return false;
    
    // Check if both are compute operators (not memory/shape ops)
    if (!a->op.as<OpNode>() || !b->op.as<OpNode>()) return false;
    
    // Simple check: don't fuse operators with large intermediate results
    // or operations that benefit from separate execution
    std::string op_a = a->op.as<OpNode>()->name;
    std::string op_b = b->op.as<OpNode>()->name;
    
    // Avoid fusing large reduction operations
    if (op_a.find("reduce") != std::string::npos || 
        op_b.find("reduce") != std::string::npos) {
      return false;
    }
    
    // Most other operations can potentially be fused
    return true;
  }
  
  void RecordFusionCandidate(const CallNode* a, const CallNode* b) {
    EdgeRecord rec;
    rec.graph_id = graph_id_;
    rec.edge_id = running_edge_counter++;
    
    // Get operator names
    rec.op_a_id = a->op.as<OpNode>()->name;
    rec.op_b_id = b->op.as<OpNode>()->name;
    
    // Get shape hashes
    rec.a_shape_hash = HashShape(a->checked_type());
    rec.b_shape_hash = HashShape(b->args[0]->checked_type());
    
    // Estimate FLOPs
    rec.flops_a = EstimateFlops(a);
    rec.flops_b = EstimateFlops(b);
    
    // Get data types
    rec.dtype_a = GetDType(a->checked_type());
    rec.dtype_b = GetDType(b->args[0]->checked_type());
    
    // These would be collected at runtime during profiling
    // For now, we use placeholders
    rec.hbm_free = 0.0f;
    rec.sm_occ = 0.0f;
    rec.chiplet_id = 0;
    rec.delta_latency_ms = 0.0f;  // Will be filled after profiling
    
    // Write to file
    std::string output_file = "dataset/edge_records/raw_edges.csv";
    SerializeToFile(rec, output_file);
  }
};

// Function to force-fuse a specific edge (used in profiling)
Function ForceFuseEdge(const Function& func, int edge_id, const std::string& graph_id) {
  // In a real implementation, this would:
  // 1. Parse the graph to find the specific edge with the given ID
  // 2. Directly apply fusion to that edge only
  // 3. Return the modified function
  
  // For this skeleton, we just return the original function
  return func;
}

// Create the adaptive fusion pass
transform::Pass CreateAdaptiveFusionPass(bool dump_edges) {
  auto pass_func = [dump_edges](IRModule mod, transform::PassContext ctx) -> IRModule {
    std::string graph_id = ctx->GetConfig("graph_id", String("unknown")).operator std::string();
    
    for (auto& kv : mod->functions) {
      if (auto* func = kv.second.as<FunctionNode>()) {
        auto new_func = AdaptiveFusionMutator(dump_edges, graph_id)
                           .Mutate(GetRef<Function>(func));
        mod->Update(kv.first, new_func);
      }
    }
    
    return mod;
  };
  
  return transform::CreateModulePass(pass_func, 0, "AdaptiveFusion", {});
}

// Register the pass with TVM
TVM_REGISTER_GLOBAL("relay.ext.adaptive_fusion")
.set_body_typed(CreateAdaptiveFusionPass);

}  // namespace relay
}  // namespace tvm 