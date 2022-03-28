#include <onnxruntime_cxx_api.h>

#ifdef DIRECTML
#include <dml_provider_factory.h>
#endif

#include <array>
#include <exception>
#include <fstream>
#include <memory>
#include <string>
#include <unordered_set>

#include "nlohmann/json.hpp"

#ifndef SHAREVOX_CORE_EXPORTS
#define SHAREVOX_CORE_EXPORTS
#endif  // SHAREVOX_CORE_EXPORTS
#include "core.h"

#define NOT_INITIALIZED_ERR "Call initialize() first."
#define NOT_FOUND_ERR "No such file or directory: "
#define FAILED_TO_OPEN_MODEL_ERR "Unable to open model files."
#define FAILED_TO_OPEN_METAS_ERR "Unable to open metas.json."
#define ONNX_ERR "ONNX raise exception: "
#define JSON_ERR "JSON parser raise exception: "
#define GPU_NOT_SUPPORTED_ERR "This library is CPU version. GPU is not supported."
#define UNKNOWN_STYLE "Unknown style ID: "

// constexpr float PHONEME_LENGTH_MINIMAL = 0.01f;
constexpr int64_t hidden_size = 256;

constexpr std::array<int64_t, 0> scalar_shape{};
constexpr std::array<int64_t, 1> speaker_shape{1};

static std::string error_message;
static bool initialized = false;
static std::string supported_devices_str;

bool open_models(const std::string variance_model_path, const std::string embedder_model_path,
                 const std::string decoder_model_path, std::vector<unsigned char> &variance_model,
                 std::vector<unsigned char> &embedder_model, std::vector<unsigned char> &decoder_model) {
  std::ifstream variance_model_file(variance_model_path, std::ios::binary),
      embedder_model_file(embedder_model_path, std::ios::binary),
      decoder_model_file(decoder_model_path, std::ios::binary);
  if (!variance_model_file.is_open() || !embedder_model_file.is_open() || !decoder_model_file.is_open()) {
    error_message = FAILED_TO_OPEN_MODEL_ERR;
    return false;
  }

  variance_model = std::vector<unsigned char>(std::istreambuf_iterator<char>(variance_model_file), {});
  embedder_model = std::vector<unsigned char>(std::istreambuf_iterator<char>(embedder_model_file), {});
  decoder_model = std::vector<unsigned char>(std::istreambuf_iterator<char>(decoder_model_file), {});
  return true;
}

/**
 * Loads the metas.json.
 *
 * schema:
 * [{
 *  name: string,
 *  styles: [{name: string, id: int}],
 *  speaker_uuid: string,
 *  version: string
 * }]
 */
bool open_metas(const std::string metas_path, nlohmann::json &metas) {
  std::ifstream metas_file(metas_path);
  if (!metas_file.is_open()) {
    error_message = FAILED_TO_OPEN_METAS_ERR;
    return false;
  }
  metas_file >> metas;
  return true;
}

struct SupportedDevices {
  bool cpu = true;
  bool cuda = false;
  bool dml = false;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(SupportedDevices, cpu, cuda, dml);

SupportedDevices get_supported_devices() {
  SupportedDevices devices;
  const auto providers = Ort::GetAvailableProviders();
  for (const std::string &p : providers) {
    if (p == "CUDAExecutionProvider") {
      devices.cuda = true;
    } else if (p == "DmlExecutionProvider") {
      devices.dml = true;
    }
  }
  return devices;
}

struct Status {
  Status(const char *root_dir_path_utf8, bool use_gpu_)
      : use_gpu(use_gpu_),
        memory_info(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU)),
        variance(nullptr),
        embedder(nullptr),
        decoder(nullptr) {
    // 扱いやすくするために、パスを正規化(スラッシュで終わるようにし、バックスラッシュもスラッシュで統一)
    std::string temp_root_dir_path(root_dir_path_utf8);
    std::vector<std::string> split_path;

    std::string item;
    for (char ch : temp_root_dir_path) {
      if (ch == '/' || ch == '\\') {
        if (!item.empty()) split_path.push_back(item);
        item.clear();
      } else {
        item += ch;
      }
    }
    if (!item.empty()) split_path.push_back(item);
    if (temp_root_dir_path[0] == '/') {
      root_dir_path = "/";
    }
    std::for_each(split_path.begin(), split_path.end(), [&](std::string path) { root_dir_path += path + "/"; });
  }

  bool load(int cpu_num_threads) {
    if (!open_metas(root_dir_path + "metas.json", metas)) {
      return false;
    }
    metas_str = metas.dump();
    supported_styles.clear();
    for (const auto &meta : metas) {
      for (const auto &style : meta["styles"]) {
        supported_styles.insert(style["id"].get<int64_t>());
      }
    }

    std::vector<unsigned char> variance_model, embedder_model, decoder_model;
    if (!open_models(root_dir_path + "variance_model.onnx", root_dir_path + "embedder_model.onnx",
                     root_dir_path + "decoder_model.onnx", variance_model, embedder_model, decoder_model)) {
      return false;
    }
    Ort::SessionOptions session_options;
    session_options.SetInterOpNumThreads(cpu_num_threads).SetIntraOpNumThreads(cpu_num_threads);
    variance = Ort::Session(env, variance_model.data(), variance_model.size(), session_options);
    embedder = Ort::Session(env, embedder_model.data(), embedder_model.size(), session_options);
    if (use_gpu) {
#ifdef DIRECTML
      session_options.DisableMemPattern().SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
      Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_DML(session_options, 0));
#else
      const OrtCUDAProviderOptions cuda_options;
      session_options.AppendExecutionProvider_CUDA(cuda_options);
#endif
    }
    decoder = Ort::Session(env, decoder_model.data(), decoder_model.size(), session_options);
    return true;
  }

  std::string root_dir_path;
  bool use_gpu;
  Ort::MemoryInfo memory_info;

  Ort::Env env{ORT_LOGGING_LEVEL_ERROR};
  Ort::Session variance, embedder, decoder;

  nlohmann::json metas;
  std::string metas_str;
  std::unordered_set<int64_t> supported_styles;
};

static std::unique_ptr<Status> status;

template <typename T, size_t Rank>
Ort::Value to_tensor(T *data, const std::array<int64_t, Rank> &shape) {
  int64_t count = 1;
  for (int64_t dim : shape) {
    count *= dim;
  }
  return Ort::Value::CreateTensor<T>(status->memory_info, data, count, shape.data(), shape.size());
}

bool validate_speaker_id(int64_t speaker_id) {
  if (status->supported_styles.find(speaker_id) == status->supported_styles.end()) {
    error_message = UNKNOWN_STYLE + std::to_string(speaker_id);
    return false;
  }
  return true;
}

bool initialize(const char *root_dir_path, bool use_gpu, int cpu_num_threads) {
  initialized = false;

#ifdef DIRECTML
  if (use_gpu && !get_supported_devices().dml) {
#else
  if (use_gpu && !get_supported_devices().cuda) {
#endif /*DIRECTML*/
    error_message = GPU_NOT_SUPPORTED_ERR;
    return false;
  }
  try {
    status = std::make_unique<Status>(root_dir_path, use_gpu);
    if (!status->load(cpu_num_threads)) {
      return false;
    }
    if (use_gpu) {
      // 一回走らせて十分なGPUメモリを確保させる
      int length = 500;
      std::vector<int64_t> phoneme(length);
      std::vector<float> pitches(length), durations(length);
      int64_t speaker_id = 0;
      std::vector<float> output(length * 256);
      decode_forward(length, phoneme.data(), pitches.data(), durations.data(), &speaker_id, output.data());
    }
  } catch (const Ort::Exception &e) {
    error_message = ONNX_ERR;
    error_message += e.what();
    return false;
  } catch (const nlohmann::json::exception &e) {
    error_message = JSON_ERR;
    error_message += e.what();
    return false;
  } catch (const std::exception &e) {
    error_message = e.what();
    return false;
  }

  initialized = true;
  return true;
}

void finalize() {
  initialized = false;
  status.reset();
}

const char *metas() { return status->metas_str.c_str(); }

const char *supported_devices() {
  SupportedDevices devices = get_supported_devices();
  nlohmann::json json = devices;
  supported_devices_str = json.dump();
  return supported_devices_str.c_str();
}

bool variance_forward(int64_t length, int64_t *phonemes, int64_t *accents, int64_t *speaker_id, float *pitch_output,
                      float *duration_output) {
  if (!initialized) {
    error_message = NOT_INITIALIZED_ERR;
    return false;
  }
  if (!validate_speaker_id(*speaker_id)) {
    return false;
  }
  try {
    const char *inputs[] = {"phonemes", "accents", "speakers"};
    const char *outputs[] = {"log_pitches", "log_durations"};
    const std::array<int64_t, 1> input_shape{length};

    std::array<Ort::Value, 3> input_tensors = {to_tensor(phonemes, input_shape), to_tensor(accents, input_shape),
                                               to_tensor(speaker_id, speaker_shape)};
    std::array<Ort::Value, 2> output_tensors = {to_tensor(pitch_output, input_shape),
                                                to_tensor(duration_output, input_shape)};

    status->variance.Run(Ort::RunOptions{nullptr}, inputs, input_tensors.data(), input_tensors.size(), outputs,
                         output_tensors.data(), output_tensors.size());

    // for (int64_t i = 0; i < length; i++) {
    //   if (pitch_output[i] < PHONEME_LENGTH_MINIMAL) pitch_output[i] = PHONEME_LENGTH_MINIMAL;
    // }
  } catch (const Ort::Exception &e) {
    error_message = ONNX_ERR;
    error_message += e.what();
    return false;
  }

  return true;
}

std::vector<float> length_regulator(int64_t length, const std::vector<float>& embedded_vector, const float *durations) {
  std::vector<float> length_regulated_vector;
  for (int64_t i = 0; i < length; i++) {
    auto regulation_size = (int64_t)(durations[i] * 256.0f);
    size_t start = length_regulated_vector.size();
    length_regulated_vector.resize(start + (regulation_size * hidden_size));
    for (int64_t j = 0; j < regulation_size * hidden_size;) {
      for (int64_t k = 0; k < hidden_size; k++) {
        length_regulated_vector[start + j + k] = embedded_vector[i + k];
      }
      j += hidden_size;
    }
  }
  return length_regulated_vector;
}

bool decode_forward(int64_t length, int64_t *phonemes, float *pitches, float *durations, int64_t *speaker_id,
                    float *output) {
  if (!initialized) {
    error_message = NOT_INITIALIZED_ERR;
    return false;
  }
  if (!validate_speaker_id(*speaker_id)) {
    return false;
  }
  try {
    const std::array<int64_t, 1> input_shape{length};
    const std::vector<float> embedded_vector(length * hidden_size);
    const std::array<int64_t, 2> embedded_shape{length, hidden_size};

    std::array<Ort::Value, 3> input_tensor = {to_tensor(phonemes, input_shape), to_tensor(pitches, input_shape),
                                              to_tensor(speaker_id, speaker_shape)};
    Ort::Value embedder_tensor = to_tensor(embedded_vector.data(), embedded_shape);
    const char *embedder_inputs[] = {"phonemes", "pitches", "speaker"};
    const char *embedder_outputs[] = {"feature_embedded"};

    status->embedder.Run(Ort::RunOptions{nullptr}, embedder_inputs, input_tensor.data(), input_tensor.size(),
                         embedder_outputs, &embedder_tensor, 1);

    std::vector<float> length_regulated_vector = length_regulator(length, embedded_vector, durations);
    const int64_t new_length = length_regulated_vector.size();
    const int64_t output_size = new_length; // hidden_size / wav frame size
    const std::array<int64_t, 2> length_regulated_shape{new_length, hidden_size};
    const std::array<int64_t, 1> wave_shape{output_size};

    Ort::Value length_regulated_tensor = to_tensor(length_regulated_vector.data(), length_regulated_shape);
    Ort::Value output_tensor = to_tensor(output, wave_shape);

    const char *decoder_inputs[] = {"feature_embedded"};
    const char *decoder_outputs[] = {"wave"};

    status->decoder.Run(Ort::RunOptions{nullptr}, decoder_inputs, &length_regulated_tensor, 1, decoder_outputs,
                        &output_tensor, 1);

  } catch (const Ort::Exception &e) {
    error_message = ONNX_ERR;
    error_message += e.what();
    return false;
  }

  return true;
}

const char *last_error_message() { return error_message.c_str(); }