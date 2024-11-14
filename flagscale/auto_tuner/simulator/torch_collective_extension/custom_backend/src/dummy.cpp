#include "dummy.hpp"
#include <iostream>
// #include <pybind11/pybind11.h>
// #include <pybind11/stl.h>
// #include <iostream>
// #include <vector>

namespace c10d {


bool WorkDummy::isCompleted() {
  return true;
}

bool WorkDummy::isSuccess() const {
  return true;
}

bool WorkDummy::wait(std::chrono::milliseconds /* unused */) {
  return true;
}

c10::intrusive_ptr<c10::ivalue::Future> WorkDummy::getFuture() {
  return future_;
}

// If necessary, pass store/rank/size to the ctor and exchange connection
// information here
BackendDummy::BackendDummy(int rank, int size)
    : Backend(rank, size) {}

// void init_process_group(
//     const std::optional<std::string>& backend = std::nullopt,
//     const std::optional<std::string>& init_method = std::nullopt,
//     const std::optional<std::chrono::milliseconds>& timeout = std::nullopt,
//     int world_size = -1,
//     int rank = -1,
//     std::optional<Store*> store = std::nullopt,
//     const std::string& group_name = "",
//     const std::optional<void*>& pg_options = std::nullopt,
//     const std::optional<torch::Device>& device_id = std::nullopt
// ) {
//   // printf("Dummy init_process_group\n");
// }


// c10::intrusive_ptr<Work> BackendDummy::_new_process_group_helper(
//   std::vector<std::vector<at::Tensor>>& outputTensors) {
//   // printf("Dummy _new_process_group_helper\n");

//   auto future = c10::make_intrusive<c10::ivalue::Future>(
//     c10::ListType::create(c10::ListType::create(c10::TensorType::get())));
//   future->markCompleted(c10::IValue(outputTensors));
//   return c10::make_intrusive<WorkDummy>(OpType::ALLGATHER, std::move(future));
// }

// This is a dummy allgather that sets all output tensors to zero
// Modify the implementation to conduct real communication asynchronously
c10::intrusive_ptr<Work> BackendDummy::allgather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const AllgatherOptions& /* unused */) {
  // printf("Dummy allgather\n");
  for (auto& outputTensorVec : outputTensors) {
      for (auto& outputTensor : outputTensorVec) {
          outputTensor.fill_(1);
      }
  }

  auto future = c10::make_intrusive<c10::ivalue::Future>(
    c10::ListType::create(c10::ListType::create(c10::TensorType::get())));
  future->markCompleted(c10::IValue(outputTensors));
  return c10::make_intrusive<WorkDummy>(OpType::ALLGATHER, std::move(future));
}

c10::intrusive_ptr<Work> BackendDummy::all_gather_object(
    std::vector<AnyType>& outputTensors,
    AnyType& inputTensors,
    const AllgatherOptions& /* unused */) {
  // printf("Dummy all_gather_object Begin\n");

  auto future = c10::make_intrusive<c10::ivalue::Future>(
    c10::ListType::create(c10::ListType::create(c10::TensorType::get())));
  return c10::make_intrusive<WorkDummy>(OpType::ALLGATHER, std::move(future));
}

c10::intrusive_ptr<Work> BackendDummy::_allgather_base(
    at::Tensor& /* unused */,
    at::Tensor& /* unused */,
    const AllgatherOptions& /* unused */) {
  // printf("Dummy _allgather_base\n");
  auto future = c10::make_intrusive<c10::ivalue::Future>(
    c10::ListType::create(c10::ListType::create(c10::TensorType::get())));
  return c10::make_intrusive<WorkDummy>(OpType::ALLGATHER, std::move(future));
}

// This is a dummy allreduce that sets all output tensors to zero
// Modify the implementation to conduct real communication asynchronously
c10::intrusive_ptr<Work> BackendDummy::allreduce(
    std::vector<at::Tensor>& tensors,
    const AllreduceOptions& opts) {
  // printf("Dummy allreduce\n");
  for (auto& tensor : tensors) {
      tensor.zero_();
  }

  auto future = c10::make_intrusive<c10::ivalue::Future>(
    c10::ListType::create(c10::TensorType::get()));
  future->markCompleted(c10::IValue(tensors));
  return c10::make_intrusive<WorkDummy>(OpType::ALLGATHER, std::move(future));
}

c10::intrusive_ptr<Work> BackendDummy::allreduce_coalesced(
    std::vector<at::Tensor>& /* unused */,
    const AllreduceCoalescedOptions& /* unused */) {
  // printf("Dummy allreduce_coalesced\n");
  auto future = c10::make_intrusive<c10::ivalue::Future>(
    c10::ListType::create(c10::ListType::create(c10::TensorType::get())));
  return c10::make_intrusive<WorkDummy>(OpType::ALLGATHER, std::move(future));
}

c10::intrusive_ptr<Work> BackendDummy::alltoall(
    std::vector<at::Tensor>& /* unused */,
    std::vector<at::Tensor>& /* unused */,
    const AllToAllOptions& /* unused */) {
  // printf("Dummy alltoall\n");
  auto future = c10::make_intrusive<c10::ivalue::Future>(
    c10::ListType::create(c10::ListType::create(c10::TensorType::get())));
  return c10::make_intrusive<WorkDummy>(OpType::ALLGATHER, std::move(future));
}

c10::intrusive_ptr<Work> BackendDummy::alltoall_base(
    at::Tensor& outputTensor,
    at::Tensor& inputTensor,
    std::vector<int64_t>& outputSplitSizes,
    std::vector<int64_t>& inputSplitSizes,
    const AllToAllOptions& /* unused */) {
  // printf("Dummy alltoall_base\n");

  auto future = c10::make_intrusive<c10::ivalue::Future>(
    c10::ListType::create(c10::ListType::create(c10::TensorType::get())));
  return c10::make_intrusive<WorkDummy>(OpType::ALLGATHER, std::move(future));
}

c10::intrusive_ptr<Work> BackendDummy::barrier(
    const BarrierOptions& /* unused */) {
  // printf("Dummy barrier\n");

  auto future = c10::make_intrusive<c10::ivalue::Future>(
    c10::ListType::create(c10::ListType::create(c10::TensorType::get())));
  return c10::make_intrusive<WorkDummy>(OpType::ALLGATHER, std::move(future));
}

c10::intrusive_ptr<Work> BackendDummy::broadcast(
    std::vector<at::Tensor>& tensors,
    const BroadcastOptions& opts) {
  // printf("Dummy broadcast\n");
  for (auto& tensor : tensors) {
      tensor.zero_();
  }

  auto future = c10::make_intrusive<c10::ivalue::Future>(
    c10::ListType::create(c10::TensorType::get()));
  future->markCompleted(c10::IValue(tensors));
  return c10::make_intrusive<WorkDummy>(OpType::ALLGATHER, std::move(future));
}

c10::intrusive_ptr<Work> BackendDummy::gather(
    std::vector<std::vector<at::Tensor>>& /* unused */,
    std::vector<at::Tensor>& /* unused */,
    const GatherOptions& /* unused */) {
  // printf("Dummy gather\n");

  auto future = c10::make_intrusive<c10::ivalue::Future>(
    c10::ListType::create(c10::ListType::create(c10::TensorType::get())));
  return c10::make_intrusive<WorkDummy>(OpType::ALLGATHER, std::move(future));
}

c10::intrusive_ptr<Work> BackendDummy::reduce(
    std::vector<at::Tensor>& /* unused */,
    const ReduceOptions& /* unused */) {
  // printf("Dummy reduce\n");

  auto future = c10::make_intrusive<c10::ivalue::Future>(
    c10::ListType::create(c10::ListType::create(c10::TensorType::get())));
  return c10::make_intrusive<WorkDummy>(OpType::ALLGATHER, std::move(future));
}

c10::intrusive_ptr<Work> BackendDummy::reduce_scatter(
    std::vector<at::Tensor>& /* unused */,
    std::vector<std::vector<at::Tensor>>& /* unused */,
    const ReduceScatterOptions& /* unused */) {
  // printf("Dummy reduce_scatter\n");
  auto future = c10::make_intrusive<c10::ivalue::Future>(
    c10::ListType::create(c10::ListType::create(c10::TensorType::get())));
  return c10::make_intrusive<WorkDummy>(OpType::ALLGATHER, std::move(future));
}

c10::intrusive_ptr<Work> BackendDummy::scatter(
    std::vector<at::Tensor>& /* unused */,
    std::vector<std::vector<at::Tensor>>& /* unused */,
    const ScatterOptions& /* unused */) {
  // printf("Dummy scatter\n");

  auto future = c10::make_intrusive<c10::ivalue::Future>(
    c10::ListType::create(c10::ListType::create(c10::TensorType::get())));
  return c10::make_intrusive<WorkDummy>(OpType::ALLGATHER, std::move(future));
}

c10::intrusive_ptr<Work> BackendDummy::send(
    std::vector<at::Tensor>& tensors,
    int dstRank,
    int tag) {
  auto future = c10::make_intrusive<c10::ivalue::Future>(
    c10::ListType::create(c10::ListType::create(c10::TensorType::get())));
  return c10::make_intrusive<WorkDummy>(OpType::ALLGATHER, std::move(future));
}

c10::intrusive_ptr<Work> BackendDummy::recv(
    std::vector<at::Tensor>& tensors,
    int srcRank,
    int tag) {
  auto future = c10::make_intrusive<c10::ivalue::Future>(
    c10::ListType::create(c10::ListType::create(c10::TensorType::get())));
  return c10::make_intrusive<WorkDummy>(OpType::ALLGATHER, std::move(future));
}

c10::intrusive_ptr<Work> BackendDummy::recvAnysource(
    std::vector<at::Tensor>& tensors,
    int tag) {
  auto future = c10::make_intrusive<c10::ivalue::Future>(
    c10::ListType::create(c10::ListType::create(c10::TensorType::get())));
  return c10::make_intrusive<WorkDummy>(OpType::ALLGATHER, std::move(future));
}

c10::intrusive_ptr<Backend> BackendDummy::createBackendDummy(
    const c10::intrusive_ptr<::c10d::Store>& /* unused */,
    int rank,
    int size,
    const std::chrono::duration<float>& /* unused */) {
  return c10::make_intrusive<BackendDummy>(rank, size);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("createBackendDummy", &BackendDummy::createBackendDummy);
}

} // namespace c10d
