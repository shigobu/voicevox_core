#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <vector>

#include "core.h"
#include "engine/kana_parser.h"
#include "engine/model.h"
#include "engine/synthesis_engine.h"

using namespace voicevox::core::engine;

static SynthesisEngine engine;

SharevoxResultCode sharevox_load_openjtalk_dict(const char *dict_path) {
  // TODO: error handling
  engine.load_openjtalk_dict(dict_path);
  return SHAREVOX_RESULT_SUCCEED;
}

SharevoxResultCode sharevox_tts(const char *text, int64_t speaker_id, int *output_binary_size,
                                uint8_t **output_wav) {
  if (!engine.is_openjtalk_dict_loaded()) {
    return SHAREVOX_RESULT_NOT_LOADED_OPENJTALK_DICT;
  }

  std::vector<AccentPhraseModel> accent_phrases = engine.create_accent_phrases(std::string(text), &speaker_id);
  const AudioQueryModel audio_query = {
      accent_phrases, 1.0f, 0.0f, 1.0f, 1.0f, 0.1f, 0.1f, engine.default_sampling_rate, false, "",
  };

  const auto wav = engine.synthesis_wave_format(audio_query, &speaker_id, output_binary_size);
  auto *wav_heap = new uint8_t[*output_binary_size];
  std::copy(wav.begin(), wav.end(), wav_heap);
  *output_wav = wav_heap;
  return SHAREVOX_RESULT_SUCCEED;
}

SharevoxResultCode sharevox_tts_from_kana(const char *text, int64_t speaker_id, int *output_binary_size,
                                          uint8_t **output_wav) {
  std::vector<AccentPhraseModel> accent_phrases = parse_kana(std::string(text));
  accent_phrases = engine.replace_mora_data(accent_phrases, &speaker_id);
  const AudioQueryModel audio_query = {
      accent_phrases, 1.0f, 0.0f, 1.0f, 1.0f, 0.1f, 0.1f, engine.default_sampling_rate, false, "",
  };

  const auto wav = engine.synthesis_wave_format(audio_query, &speaker_id, output_binary_size);
  auto *wav_heap = new uint8_t[*output_binary_size];
  std::copy(wav.begin(), wav.end(), wav_heap);
  *output_wav = wav_heap;
  return SHAREVOX_RESULT_SUCCEED;
}

void sharevox_wav_free(uint8_t *wav) { delete wav; }

const char *sharevox_error_result_to_message(SharevoxResultCode result_code) {
  switch (result_code) {
    case SHAREVOX_RESULT_NOT_LOADED_OPENJTALK_DICT:
      return "Call sharevox_load_openjtalk_dict() first.";

    default:
      throw std::runtime_error("Unexpected error result code.");
  }
}
