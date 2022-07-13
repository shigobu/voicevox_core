#pragma once

#ifdef _WIN32
#ifdef SHAREVOX_CORE_EXPORTS
#define SHAREVOX_CORE_API __declspec(dllexport)
#else  // SHAREVOX_CORE_EXPORTS
#define SHAREVOX_CORE_API __declspec(dllimport)
#endif  // SHAREVOX_CORE_EXPORTS
#else   // _WIN32
#define SHAREVOX_CORE_API
#endif  // _WIN32

#ifdef __cplusplus
#include <cstdint>
#else
#include <stdint.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @enum
 * 結果コード
 * エラーの種類が増えたら定義を増やす。
 * 必ずエラーの値を明示的に指定すること
 */
typedef enum {
  // 成功
  SHAREVOX_RESULT_SUCCEED = 0,
  // OpenJTalk辞書がロードされていない
  SHAREVOX_RESULT_NOT_LOADED_OPENJTALK_DICT = 1,
} SharevoxResultCode;
/**
 * @fn
 * 初期化する
 * @brief 音声合成するための初期化を行う。他の関数を正しく実行するには先に初期化が必要
 * @param root_dir_path 必要なファイルがあるディレクトリ。相対パス・絶対パスどちらも指定可能。文字コードはUTF-8
 * @param use_gpu trueならGPU用、falseならCPU用の初期化を行う
 * @param cpu_num_threads 推論に用いるスレッド数を設定する。0の場合論理コア数の半分か、物理コア数が設定される
 * @param load_all_models trueなら全てのモデルをロードする
 * @return 成功したらtrue、失敗したらfalse
 * @detail
 * 何度も実行可能。use_gpuを変更して実行しなおすことも可能。
 * 最後に実行したuse_gpuに従って他の関数が実行される。
 */
SHAREVOX_CORE_API bool initialize(const char *root_dir_path, bool use_gpu,
                                  int cpu_num_threads
#ifdef __cplusplus
                                  = 0
#endif
                                  ,
                                  bool load_all_models
#ifdef __cplusplus
                                  = true
#endif
);

/**
 * モデルをロードする
 * @param speaker_id 話者番号
 * @return 成功したらtrue、失敗したらfalse
 * @detail
 * 必ずしも話者とモデルが1:1対応しているわけではない。
 */
SHAREVOX_CORE_API bool load_model(const char *speaker_id);

/**
 * @fn
 * モデルがロード済みかどうか
 * @param speaker_id 話者番号
 * @return ロード済みならtrue、そうでないならfalse
 */
SHAREVOX_CORE_API bool is_model_loaded(const char *speaker_id);

/**
 * @fn
 * 終了処理を行う
 * @brief 終了処理を行う。以降関数を利用するためには再度初期化を行う必要がある。
 * @detail
 * 何度も実行可能。実行せずにexitしても大抵の場合問題ないが、
 * CUDAを利用している場合これを実行しておかないと例外が起こることがある。
 */
SHAREVOX_CORE_API void finalize();

/**
 * @fn
 * メタ情報を取得する
 * @brief 話者名や話者IDのリストを取得する
 * @return メタ情報が格納されたjson形式の文字列
 */
SHAREVOX_CORE_API const char *metas();

/**
 * @fn
 * 対応デバイス情報を取得する
 * @brief cpu, cudaのうち、使用可能なデバイス情報を取得する
 * @return 各デバイスが使用可能かどうかをboolで格納したjson形式の文字列
 */
SHAREVOX_CORE_API const char *supported_devices();

/**
 * @fn
 * 音素ごとの音高と長さを求める
 * @brief 音素列とアクセント列から、音素ごとの音高と長さを求める
 * @param length 音素列・アクセント列の長さ
 * @param phonemes 音素列
 * @param accents アクセント列
 * @param speaker_id 話者番号
 * @return 音素ごとの長さ・モーラごとの音高
 */
SHAREVOX_CORE_API bool variance_forward(int64_t length, int64_t *phonemes, int64_t *accents, const char *speaker_id,
                                        float *pitch_output, float *duration_output);

/**
 * @fn
 * 波形を求める
 * @brief 音素と音素ごとの音高・長さから、波形を求める
 * @param length 音素列の長さ
 * @param phonemes 音素列
 * @param pitches 音素ごとの音高
 * @param durations 音素ごとの長さ
 * @param speaker_id 話者番号
 * @return 音声波形
 */
SHAREVOX_CORE_API bool decode_forward(int64_t length, int64_t *phonemes, float *pitches, float *durations,
                                      const char *speaker_id, float *output);

/**
 * @fn
 * 最後に発生したエラーのメッセージを取得する
 * @return エラーメッセージ
 */
SHAREVOX_CORE_API const char *last_error_message();

/**
 * @fn
 * open jtalkの辞書を読み込む
 * @return 結果コード
 */
SHAREVOX_CORE_API SharevoxResultCode sharevox_load_openjtalk_dict(const char *dict_path);

/**
 * @fn
 * text to spearchを実行する
 * @param text 音声データに変換するtextデータ
 * @param speaker_id 話者番号
 * @param output_binary_size 音声データのサイズを出力する先のポインタ
 * @param output_wav 音声データを出力する先のポインタ。使用が終わったらvoicevox_wav_freeで開放する必要がある
 * @return 結果コード
 */
SHAREVOX_CORE_API SharevoxResultCode sharevox_tts(const char *text, const char *speaker_id, int *output_binary_size,
                                                  uint8_t **output_wav);

/**
 * @fn
 * text to spearchをAquesTalkライクな記法で実行する
 * @param text 音声データに変換するtextデータ
 * @param speaker_id 話者番号
 * @param output_binary_size 音声データのサイズを出力する先のポインタ
 * @param output_wav 音声データを出力する先のポインタ。使用が終わったらvoicevox_wav_freeで開放する必要がある
 * @return 結果コード
 */
SHAREVOX_CORE_API SharevoxResultCode sharevox_tts_from_kana(const char *text, const char *speaker_id,
                                                            int *output_binary_size, uint8_t **output_wav);

/**
 * @fn
 * voicevox_ttsで生成した音声データを開放する
 * @param wav 開放する音声データのポインタ
 */
SHAREVOX_CORE_API void sharevox_wav_free(uint8_t *wav);

/**
 * @fn
 * エラーで返ってきた結果コードをメッセージに変換する
 * @return エラーメッセージ文字列
 */
SHAREVOX_CORE_API const char *sharevox_error_result_to_message(SharevoxResultCode result_code);

#ifdef __cplusplus
}
#endif

// 使い終わったマクロ定義は不要なので解除する
#undef SHAREVOX_CORE_API
