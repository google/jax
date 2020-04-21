/* Copyright 2020 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "absl/container/flat_hash_map.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "include/pybind11/pybind11.h"
#include "include/pybind11/stl.h"
#include "include/pybind11/pytypes.h"
#include "jaxlib/kernel_pybind11_helpers.h"
#include "include/msgpack.hpp"

namespace {

namespace py = pybind11;

// The Python code will query the version of the metadata, and
// it can abort or encode specially for backwards compatibility.
// Bump this version always when you change the metadata format,
// even if it is backwards compatible.
// As long as we use msgpack, the client can add new objects at
// the end while being compatible with the library.
int constexpr kPrintMetadataVersion = 1;

int GetPrintMetadataVersion() {
  return kPrintMetadataVersion;
}

// Metadata for id_print runtime functions.
typedef std::vector<int> Shape;
enum ElementType {
  I8, I16, I32, I64,
  U8, U16, U32, U64,
  F16, F32, F64,
};

struct TypeAndShape {
  ElementType element_type;
  size_t element_size;
  Shape shape;
};
struct PrintMetadata {
  // The preamble to be printed before the arguments.
  std::string preamble;
  // The separator to be printed between the arguments.
  std::string separator;
  // Types and shapes for the arguments to be printed.
  std::vector<TypeAndShape> args_type_and_shape;
};

// Converts a type descriptor and shape to TypeAndShape.
TypeAndShape ParseTypeDescriptor(const std::tuple<std::string, Shape>& type_and_shape) {
  static auto* types = new absl::flat_hash_map<std::pair<char, int>, ElementType>({
      {{'f', 2}, ElementType::F16},
      {{'f', 4}, ElementType::F32},
      {{'f', 8}, ElementType::F64},
      {{'i', 1}, ElementType::I8},
      {{'i', 2}, ElementType::I16},
      {{'i', 4}, ElementType::I32},
      {{'i', 8}, ElementType::I64},
      {{'u', 1}, ElementType::U8},
      {{'u', 2}, ElementType::U16},
      {{'u', 4}, ElementType::U32},
      //{{'u', 8}, ElementType::U64},
  });
  std::string type_descriptor = std::get<0>(type_and_shape);
  size_t element_size;
  if (!absl::SimpleAtoi(type_descriptor.substr(1), &element_size)) {
    throw std::invalid_argument(
        absl::StrFormat("Unsupported type descriptor %s (no size found)", type_descriptor));
  }

  auto it = types->find({type_descriptor.at(0), element_size});
  if (it == types->end()) {
    throw std::invalid_argument(
        absl::StrFormat("Unsupported type descriptor %s", type_descriptor));
  }
  return TypeAndShape{it->second, element_size, std::get<1>(type_and_shape)};
}

// Parses PrintMetadata msgpack-encoded by Python.
// The metadata has the following format:
//     (preamble: str,    # to be printed before the first argument
//      separator: str,   # to be printed between arguments
//      [ (type_descriptor: str,
//         shape: Tuple[int, ...]) ]
//
PrintMetadata ParsePrintMetadata(std::string bytes) {
  const char* buffer = bytes.data();
  const size_t len = bytes.size();
  msgpack::object_handle oh; // deserialized object is valid while handle is alive.

  PrintMetadata meta;
  size_t offset = 0;
  unpack(oh, buffer, len, offset);  // Updates oh and offset
  meta.preamble = oh.get().as<std::string>();
  unpack(oh, buffer, len, offset);
  meta.separator = oh.get().as<std::string>();

  unpack(oh, buffer, len, offset);
  auto res = oh.get().as<std::vector<std::tuple<std::string, Shape>>>();
  for (auto const &t_and_s : res) {
    meta.args_type_and_shape.push_back(ParseTypeDescriptor(t_and_s));
  }
  return meta;
}

// TODO(necula): add parameters for these
int constexpr kSideElements = 3;
int constexpr kSummarizeThreshold = 100;
int constexpr kPrecision = 2;  // Decimal digits

// TODO(necula): add unit tests
class Printer {
public:
  Printer(std::ostringstream &output, // TODO: pointer
          const TypeAndShape &type_and_shape, const uint8_t* data) :
        output_{output}, type_and_shape_(type_and_shape), current_ptr_{data},
        shape_{type_and_shape.shape},
        element_size_{type_and_shape.element_size} {
    ndims_ = type_and_shape.shape.size();
    current_index_.reserve(ndims_);
    skip_values_.reserve(ndims_);
    int current_skip = 1;
    for (int i = ndims_ - 1; i >= 0; --i) {
      current_index_[i] = 0;
      skip_values_[i] = current_skip;
      current_skip *= shape_[i];
    }
    total_size_ = current_skip;
  }

  void EmitArray();

private:
  std::ostringstream &output_;
  TypeAndShape type_and_shape_;
  const uint8_t* current_ptr_;

  Shape shape_;  // TODO(add accessors for these?)

  size_t element_size_;
  int ndims_;
  size_t total_size_;

  // The current index to be emitted: [i0, i1, ..., in-1].
  Shape current_index_;
  // For each dimension, how many elements to skip to get
  // to the next value in the same dimension.
  Shape skip_values_;

  void EmitInnermostDimension();
  void EmitCurrentElement();
};

// Emits the element at current_ptr.
void Printer::EmitCurrentElement() {
  switch (type_and_shape_.element_type) {
    case I8:
      output_ << *reinterpret_cast<const int8_t*>(current_ptr_);
      break;
    case I16:
      output_ << *reinterpret_cast<const int16_t*>(current_ptr_);
      break;
    case I32:
      output_ << *reinterpret_cast<const int32_t*>(current_ptr_);
      break;
    case I64:
      output_ << *reinterpret_cast<const int64_t*>(current_ptr_);
      break;
    case U8:
      output_ << *reinterpret_cast<const uint8_t*>(current_ptr_);
      break;
    case U16:
      output_ << *reinterpret_cast<const uint16_t*>(current_ptr_);
      break;
    case U32:
      output_ << *reinterpret_cast<const uint32_t*>(current_ptr_);
      break;
    case U64:
      output_ << *reinterpret_cast<const uint64_t*>(current_ptr_);
      break;
    case F16:
      output_ << *reinterpret_cast<const uint16_t*>(current_ptr_); // TODO(float16)
      break;
    case F32:
      output_ << *reinterpret_cast<const float*>(current_ptr_);
      break;
    case F64:
      output_ << *reinterpret_cast<const double*>(current_ptr_);
      break;
  }
}

// Emits spaces and [, then the elements in the current
// innermost dimension, then ].
// Assumes current_index[ndims - 1] = 0, current_ptr points to first
// element in the dimension to be printed.
void Printer::EmitInnermostDimension() {
  // Emit ndim spaces and [. As many [ as there are trailing 0s in current_index.
  assert(current_index_[ndims_ - 1] == 0);
  int count_start_spaces = ndims_ - 1;
  while (count_start_spaces >= 1 && current_index_[count_start_spaces - 1] == 0) {
    --count_start_spaces;
  }
  for (int i = 0; i < ndims_; ++i) {
    output_ << (i < count_start_spaces ? ' ' : '[');
  }
  // Now emit the elements
  for (int idx = 0; idx < shape_[ndims_ - 1]; ++idx, current_ptr_ += element_size_) {
    EmitCurrentElement();
    if (idx < shape_[ndims_ - 1] - 1)
      output_ << ' ';
    if (total_size_ > kSummarizeThreshold &&
        shape_[ndims_ - 1] > 2 * kSideElements &&
        idx == kSideElements - 1) {
      int skip_indices = shape_[ndims_ - 1] - kSideElements - 1 - idx;
      current_ptr_ += element_size_ * skip_indices;
      idx += skip_indices;
      output_ << "... ";
    }
  }
  // Update the index to last one emitted.
  current_index_[ndims_ - 1] = shape_[ndims_ - 1] - 1;
  int count_stop_brackets = 0;
  // Emit as many ] as how many inner dimensions have reached the end
  for (int i = ndims_ - 1;
       i >= 0 && current_index_[i] == shape_[i] - 1;
       --i) {
    ++count_stop_brackets;
    output_ << ']';
  }
  if (count_stop_brackets > 1) {
    output_ << std::endl;
  }
  output_ << std::endl;
}

// Emits a string representation of the array to output,
// in the style of numpy.array2string.
void Printer::EmitArray() {
  output_.precision(kPrecision);
  output_.setf(std::ios::fixed);
  if (ndims_ == 0) {
    EmitCurrentElement();
    return;
  }
  while (true) {
    EmitInnermostDimension();
    assert (current_index_[ndims_ - 1] == shape_[ndims_ - 1] - 1);

    // Advance to the next innermost dimension
    int dim_to_advance = ndims_ - 1;
    for( ; dim_to_advance >= 0; --dim_to_advance) {
      ++ current_index_[dim_to_advance];
      if(current_index_[dim_to_advance] >= shape_[dim_to_advance]) {
        current_index_[dim_to_advance] = 0;
        continue;
      } else {
        // Have not reached the end of the dim_to_advance.
        if (total_size_ > kSummarizeThreshold &&
            current_index_[dim_to_advance] == kSideElements &&
            shape_[dim_to_advance] > 2 * kSideElements) {
          int skip_indices = shape_[dim_to_advance] - kSideElements - current_index_[dim_to_advance];
          current_ptr_ += element_size_ * skip_values_[dim_to_advance] * skip_indices;
          current_index_[dim_to_advance] += skip_indices;
          for (int j = 0; j <= dim_to_advance; ++j) {
            output_ << ' ';
          }
          output_ << "..." << std::endl;
        }
        break;
      }
    }
    if (dim_to_advance < 0) {
      return;
    }
  }
}

void DoPrint(const PrintMetadata &meta, std::vector<const void*> args) {
  std::ostringstream oss;
  oss << meta.preamble << "\n";
  for (int i = 0; i < args.size(); ++i) {
    const TypeAndShape &arg_type_and_shape = meta.args_type_and_shape[i];
    oss << absl::StreamFormat("arg[%d] ", i);
    oss << " shape = (";
    for (const int &dim : arg_type_and_shape.shape) {
      oss << dim << ", ";
    }
    oss << ")\n";

    Printer printer(oss, arg_type_and_shape, reinterpret_cast<const uint8_t*>(args[i]));
    printer.EmitArray();
    if (i < args.size() - 1)
        oss << meta.separator;
  }
  std::cout << oss.str();
}

// Prints the arguments and returns True.
//
void PrintCPU(void* out, const void** args) {
  static constexpr int kReservedArgs = 2;
  const int* opaque_len = static_cast<const int*>(args[0]);
  const char* opaque = static_cast<const char*>(args[1]);
  const PrintMetadata& meta = ParsePrintMetadata(std::string(opaque, *opaque_len));

  std::vector<const void*> args_vector(meta.args_type_and_shape.size());
  for (int i = 0; i < meta.args_type_and_shape.size(); i++) {
    args_vector[i] = args[kReservedArgs + i];
  }
  DoPrint(meta, args_vector);

  bool *resultPtr = static_cast<bool *>(out);
  *resultPtr = true;
}

}   // namespace

namespace jax {

// Returns a dictionary with CustomCall functions to register for CPU.
py::dict CustomCallRegistrations() {
  py::dict dict;
  dict["jax_print_cpu"] = EncapsulateFunction(PrintCPU);
  return dict;
}

PYBIND11_MODULE(host_callback, m) {
  m.doc() = "Python bindings for the host_callback runtime";
  m.def("customcall_registrations", &CustomCallRegistrations);
  m.def("get_print_metadata_version", &GetPrintMetadataVersion);
}


}  // namespace jax