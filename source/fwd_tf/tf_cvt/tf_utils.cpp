#include "tf_utils.h"

FWD_TF_NAMESPACE_BEGIN

float RandomFloat() {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  static std::uniform_real_distribution<float> dis;
  return dis(gen);
}

int RandomInt() {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  static std::uniform_int_distribution<int> dis;
  return dis(gen);
}

std::shared_ptr<TF_Session> CreateSession(
    const Graph& graph, std::shared_ptr<TF_SessionOptions> options /*= nullptr*/) {
  Status status;

  if (options == nullptr) {
    options = std::shared_ptr<TF_SessionOptions>(TF_NewSessionOptions(), TF_DeleteSessionOptions);
  }

  std::shared_ptr<TF_Session> session(TF_NewSession(graph.get(), options.get(), status),
                                      DeleteSession);
  if (!status.Ok()) {
    return nullptr;
  }

  return session;
}

TF_Code DeleteSession(TF_Session* session) {
  if (session == nullptr) {
    return TF_INVALID_ARGUMENT;
  }

  Status status;

  TF_CloseSession(session, status);
  if (!status.Ok()) {
    return status.Code();
  }

  TF_DeleteSession(session, status);
  if (!status.Ok()) {
    return status.Code();
  }

  return TF_OK;
}


TF_Buffer* ReadBufferFromFile(const std::string& filename) {
  std::ifstream f(filename, std::ios::binary);
  if (f.fail() || !f.is_open()) {
    return nullptr;
  }

  if (f.seekg(0, std::ios::end).fail()) {
    return nullptr;
  }
  auto fsize = f.tellg();
  if (f.seekg(0, std::ios::beg).fail()) {
    return nullptr;
  }

  if (fsize <= 0) {
    return nullptr;
  }

  auto data = static_cast<char*>(std::malloc(fsize));
  if (f.read(data, fsize).fail()) {
    free(data);
    return nullptr;
  }

  auto buf = TF_NewBuffer();
  buf->data = data;
  buf->length = fsize;
  buf->data_deallocator = DeallocateBuffer;

  f.close();
  return buf;
}

std::shared_ptr<TF_Tensor> CreateEmptyTensor(TF_DataType data_type,
                                             const std::vector<int64_t>& dims) {
  const size_t size = std::accumulate(dims.begin(), dims.end(), 1ll, std::multiplies<int64_t>());
  const auto data_len = size * GetElementSize(data_type);

  auto tensor = std::shared_ptr<TF_Tensor>(
      TF_AllocateTensor(data_type, dims.data(), static_cast<int>(dims.size()), data_len),
      TF_DeleteTensor);

  if (tensor == nullptr) {
    return {};
  }
  return tensor;
}

FWD_TF_NAMESPACE_END