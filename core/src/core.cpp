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
#include <iostream>

#include "nlohmann/json.hpp"

#ifndef SHAREVOX_CORE_EXPORTS
#define SHAREVOX_CORE_EXPORTS
#endif  // SHAREVOX_CORE_EXPORTS
#include "core.h"

#define NOT_INITIALIZED_ERR "Call initialize() first."
#define NOT_FOUND_ERR "No such file or directory: "
#define FAILED_TO_OPEN_MODEL_ERR "Unable to open model files."
#define FAILED_TO_OPEN_METAS_ERR "Unable to open metas.json."
#define FAILED_TO_OPEN_LIBRARIES_ERR "Unable to open libraries.json."
#define NOT_LOADED_ERR "Model is not loaded."
#define ONNX_ERR "ONNX raise exception: "
#define JSON_ERR "JSON parser raise exception: "
#define GPU_NOT_SUPPORTED_ERR "This library is CPU version. GPU is not supported."
#define UNKNOWN_STYLE "Unknown style ID: "

// constexpr float PHONEME_LENGTH_MINIMAL = 0.01f;
constexpr int64_t hidden_size = 192;

constexpr std::array<int64_t, 0> scalar_shape{};
constexpr std::array<int64_t, 1> speaker_shape{1};

static std::string error_message;
static bool initialized = false;
static std::string supported_devices_str;

struct ModelData {
  std::vector<unsigned char> variance;
  std::vector<unsigned char> embedder;
  std::vector<unsigned char> decoder;
};

struct Models {
  Ort::Session variance;
  Ort::Session embedder;
  Ort::Session decoder;
};

bool open_model_files(const std::string &root_dir_path, const std::string library_uuid,
                      std::vector<unsigned char> &variance_model, std::vector<unsigned char> &embedder_model,
                      std::vector<unsigned char> &decoder_model) {
  const std::string variance_model_path = root_dir_path + library_uuid + "/variance_model.onnx";
  const std::string embedder_model_path = root_dir_path + library_uuid + "/embedder_model.onnx";
  const std::string decoder_model_path = root_dir_path + library_uuid + "/decoder_model.onnx";
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
bool open_metas(const std::string root_dir_path, const std::string library_uuid, nlohmann::json &metas) {
  std::ifstream metas_file(root_dir_path + library_uuid + "/metas.json");
  if (!metas_file.is_open()) {
    error_message = FAILED_TO_OPEN_METAS_ERR;
    return false;
  }
  metas_file >> metas;
  return true;
}

bool open_libraries(const std::string root_dir_path, nlohmann::json &libraries) {
  std::string libraries_path = root_dir_path + "libraries.json";
  std::ifstream libraries_file(libraries_path);
  if (!libraries_file.is_open()) {
    error_message = FAILED_TO_OPEN_LIBRARIES_ERR;
    return false;
  }
  libraries_file >> libraries;
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
  Status(const char *root_dir_path_utf8, bool use_gpu, int cpu_num_threads)
      : memory_info(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU)) {
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

    // 軽いモデルの場合はCPUの方が速い
    light_session_options.SetInterOpNumThreads(cpu_num_threads).SetIntraOpNumThreads(cpu_num_threads);

    // 重いモデルはGPUを使ったほうが速い
    heavy_session_options.SetInterOpNumThreads(cpu_num_threads).SetIntraOpNumThreads(cpu_num_threads);
    if (use_gpu) {
#ifdef DIRECTML
      heavy_session_options.DisableMemPattern().SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
      Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_DML(heavy_session_options, 0));
#else
      const OrtCUDAProviderOptions cuda_options;
      heavy_session_options.AppendExecutionProvider_CUDA(cuda_options);
#endif
    }
  }

  bool load() {
    if (!open_libraries(root_dir_path, libraries)) {
      return false;
    }
    libraries_str = libraries.dump();
    usable_libraries.clear();
    for (auto &library : libraries.items()) {
      if (library.value().get<bool>()) {
        usable_libraries.insert(library.key());
      }
    }

    nlohmann::json all_metas;
    for (const std::string &library_uuid : usable_libraries) {
      std::vector<unsigned char> variance_model, embedder_model, decoder_model;
      if (!open_model_files(root_dir_path, library_uuid, variance_model, embedder_model, decoder_model)) {
        return false;
      }
      nlohmann::json metas;
      if (!open_metas(root_dir_path, library_uuid, metas)) {
        return false;
      }

      usable_model_data_map.insert(std::make_pair(library_uuid, ModelData{
                                                                    std::move(variance_model),
                                                                    std::move(embedder_model),
                                                                    std::move(decoder_model),
                                                                }));

      for (auto &meta : metas) {
        for (auto &style : meta["styles"]) {
          speaker_id_map.insert(std::make_pair(style["id"].get<int64_t>(), library_uuid));
        }
        all_metas.push_back(meta);
      }
    }
    metas_str = all_metas.dump();
    return true;
  }

  void load_model(const std::string &library_uuid) {
    ModelData model_data = usable_model_data_map.at(library_uuid);
    auto variance = Ort::Session(env, model_data.variance.data(), model_data.variance.size(), light_session_options);
    auto embedder = Ort::Session(env, model_data.embedder.data(), model_data.embedder.size(), light_session_options);
    auto decoder = Ort::Session(env, model_data.decoder.data(), model_data.decoder.size(), heavy_session_options);

    usable_model_map.insert(std::make_pair(library_uuid, Models{
                                                             std::move(variance),
                                                             std::move(embedder),
                                                             std::move(decoder),
                                                         }));
    usable_model_data_map.erase(library_uuid);
  }

  std::string root_dir_path;
  Ort::SessionOptions light_session_options;  // 軽いモデルはこちらを使う
  Ort::SessionOptions heavy_session_options;  // 重いモデルはこちらを使う
  Ort::MemoryInfo memory_info;

  Ort::Env env{ORT_LOGGING_LEVEL_ERROR};

  nlohmann::json libraries;
  std::string libraries_str;
  std::string metas_str;
  std::unordered_set<std::string> usable_libraries;
  std::map<int64_t, std::string> speaker_id_map;
  std::map<std::string, ModelData> usable_model_data_map;
  std::map<std::string, Models> usable_model_map;
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

/**
 * 複数モデルのspeaker_idマッピング
 */
std::string get_library_uuid_from_speaker_id(int64_t speaker_id) {
  const auto found = status->speaker_id_map.find(speaker_id);
  if (found == status->speaker_id_map.end()) {
    return "";
  }
  return found->second;
}


bool initialize(const char *root_dir_path, bool use_gpu, int cpu_num_threads, bool load_all_models) {
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
    status = std::make_unique<Status>(root_dir_path, use_gpu, cpu_num_threads);
    if (!status->load()) {
      return false;
    }
    if (load_all_models) {
      for (const std::string &library_uuid : status->usable_libraries) {
        status->load_model(library_uuid);
      }
    }
    // officialモデルが常にあるとは限らない、0番の情報がmetasにあるとは限らないので実行しない
    // if (use_gpu) {
    //   // 一回走らせて十分なGPUメモリを確保させる
    //   int length = 500;
    //   std::vector<int64_t> phoneme(length);
    //   std::vector<float> pitches(length), durations(length);
    //   std::string speaker_id = "official_0";
    //   std::vector<float> output(length * 256);
    //   decode_forward(length, phoneme.data(), pitches.data(), durations.data(), speaker_id.c_str(), output.data());
    // }
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

bool load_model(int64_t speaker_id) {
  try {
    auto library_uuid = get_library_uuid_from_speaker_id(speaker_id);
    std::cout << library_uuid << ": " << speaker_id << std::endl;
    if (library_uuid.empty()) {
      error_message = UNKNOWN_STYLE + std::to_string(speaker_id);
      return false;
    }
    status->load_model(library_uuid);
    return true;
  } catch (std::runtime_error &e) {
    error_message = e.what();
    return false;
  } catch (const Ort::Exception &e) {
    error_message = ONNX_ERR;
    error_message += e.what();
    return false;
  }
}

bool is_model_loaded(int64_t speaker_id) {
  try {
    auto library_uuid = get_library_uuid_from_speaker_id(speaker_id);
    return status->usable_model_map.count(library_uuid) > 0;
  } catch (std::runtime_error &e) {
    error_message = e.what();
    return false;
  }
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
  std::string library_uuid = get_library_uuid_from_speaker_id(*speaker_id);
  if (library_uuid.empty()) {
    error_message = UNKNOWN_STYLE + std::to_string(*speaker_id);
    return false;
  }

  if (!is_model_loaded(*speaker_id)) {
    error_message = NOT_LOADED_ERR;
    return false;
  }

  try {
    const char *inputs[] = {"phonemes", "accents", "speakers"};
    const char *outputs[] = {"pitches", "durations"};
    const std::array<int64_t, 2> input_shape{1, length};
    const std::array<int64_t, 3> output_shape{1, length, 1};

    std::array<Ort::Value, 3> input_tensors = {to_tensor(phonemes, input_shape), to_tensor(accents, input_shape),
                                               to_tensor(speaker_id, speaker_shape)};
    std::array<Ort::Value, 2> output_tensors = {to_tensor(pitch_output, output_shape),
                                                to_tensor(duration_output, output_shape)};

    status->usable_model_map.at(library_uuid)
        .variance.Run(Ort::RunOptions{nullptr}, inputs, input_tensors.data(), input_tensors.size(), outputs,
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

std::vector<float> length_regulator(int64_t length, const std::vector<float> &embedded_vector, const float *durations) {
  std::vector<float> length_regulated_vector;
  for (int64_t i = 0; i < length; i++) {
    auto regulation_size = (int64_t)(durations[i] * 187.5);  // 48000 / 256 = 187.5
    size_t start = length_regulated_vector.size();
    length_regulated_vector.resize(start + (regulation_size * hidden_size));
    for (int64_t j = 0; j < regulation_size * hidden_size; j += hidden_size) {
      for (int64_t k = 0; k < hidden_size; k++) {
        length_regulated_vector[start + j + k] = embedded_vector[i * hidden_size + k];
      }
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
  std::string library_uuid = get_library_uuid_from_speaker_id(*speaker_id);
  if (library_uuid.empty()) {
    error_message = UNKNOWN_STYLE + std::to_string(*speaker_id);
    return false;
  }

  if (!is_model_loaded(*speaker_id)) {
    error_message = NOT_LOADED_ERR;
    return false;
  }

  try {
    const std::array<int64_t, 2> input_shape{1, length};
    std::vector<float> embedded_vector(length * hidden_size);
    const std::array<int64_t, 3> embedded_shape{1, length, hidden_size};

    std::array<Ort::Value, 3> input_tensor = {to_tensor(phonemes, input_shape), to_tensor(pitches, input_shape),
                                              to_tensor(speaker_id, speaker_shape)};
    Ort::Value embedder_tensor = to_tensor(embedded_vector.data(), embedded_shape);
    const char *embedder_inputs[] = {"phonemes", "pitches", "speakers"};
    const char *embedder_outputs[] = {"feature_embedded"};

    status->usable_model_map.at(library_uuid)
        .embedder.Run(Ort::RunOptions{nullptr}, embedder_inputs, input_tensor.data(), input_tensor.size(),
                      embedder_outputs, &embedder_tensor, 1);

    std::vector<float> length_regulated_vector = length_regulator(length, embedded_vector, durations);
    const int64_t new_length = length_regulated_vector.size() / hidden_size;
    const int64_t output_size = new_length * 256;
    const std::array<int64_t, 3> length_regulated_shape{1, new_length, hidden_size};
    const std::array<int64_t, 2> wave_shape{1, output_size};

    Ort::Value length_regulated_tensor = to_tensor(length_regulated_vector.data(), length_regulated_shape);
    Ort::Value output_tensor = to_tensor(output, wave_shape);

    const char *decoder_inputs[] = {"length_regulated_tensor"};
    const char *decoder_outputs[] = {"wav"};

    status->usable_model_map.at(library_uuid)
        .decoder.Run(Ort::RunOptions{nullptr}, decoder_inputs, &length_regulated_tensor, 1, decoder_outputs,
                     &output_tensor, 1);

  } catch (const Ort::Exception &e) {
    error_message = ONNX_ERR;
    error_message += e.what();
    return false;
  }

  return true;
}

const char *last_error_message() { return error_message.c_str(); }