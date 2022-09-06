from ctypes import *
import platform
import os
from pathlib import Path
from typing import Tuple
import numpy

# numpy ndarray types
int64_dim1_type = numpy.ctypeslib.ndpointer(dtype=numpy.int64, ndim=1)
float32_dim1_type = numpy.ctypeslib.ndpointer(dtype=numpy.float32, ndim=1)
int64_dim2_type = numpy.ctypeslib.ndpointer(dtype=numpy.int64, ndim=2)
float32_dim2_type = numpy.ctypeslib.ndpointer(dtype=numpy.float32, ndim=2)
array = numpy.ndarray


get_os = platform.system()

lib_file = ""
if get_os == "Windows":
    lib_file = "core.dll"
elif get_os == "Darwin":
    lib_file = "libcore.dylib"
elif get_os == "Linux":
    lib_file = "libcore.so"

# ライブラリ読み込み
core_dll_path = Path(os.path.dirname(__file__) + f"/lib/{lib_file}")
if not os.path.exists(core_dll_path):
    raise Exception(f"coreライブラリファイルが{core_dll_path}に存在しません")
lib = cdll.LoadLibrary(str(core_dll_path))

# 関数型定義
lib.initialize.argtypes = (c_char_p, c_bool, c_int, c_bool)
lib.initialize.restype = c_bool

lib.load_model.argtypes = (c_int64,)
lib.load_model.restype = c_bool

lib.is_model_loaded.argtypes = (c_int64,)
lib.is_model_loaded.restype = c_bool

lib.finalize.argtypes = ()

lib.metas.restype = c_char_p

lib.supported_devices.restype = c_char_p

lib.variance_forward.argtypes = (
    c_int64, int64_dim1_type, int64_dim1_type, int64_dim1_type, float32_dim1_type, float32_dim1_type)
lib.variance_forward.restype = c_bool

lib.decode_forward.argtypes = (
    c_int64, int64_dim1_type, float32_dim1_type, float32_dim1_type, int64_dim1_type, float32_dim1_type)
lib.decode_forward.restype = c_bool

lib.last_error_message.restype = c_char_p

lib.sharevox_load_openjtalk_dict.argtypes = (c_char_p,)
lib.sharevox_load_openjtalk_dict.restype = c_int

lib.sharevox_tts.argtypes = (c_char_p, c_int64, POINTER(c_int), POINTER(POINTER(c_uint8)))
lib.sharevox_tts.restype = c_int

lib.sharevox_tts_from_kana.argtypes = (c_char_p, c_int64, POINTER(c_int), POINTER(POINTER(c_uint8)))
lib.sharevox_tts_from_kana.restype = c_int

lib.sharevox_wav_free.argtypes = (POINTER(c_uint8),)

lib.sharevox_error_result_to_message.argtypes = (c_int,)
lib.sharevox_load_openjtalk_dict.argtypes = (c_char_p,)

default_sampling_rate = 48000


# ラッパー関数
def initialize(root_dir_path: str, use_gpu: bool, cpu_num_threads=0, load_all_models=True):
    path = create_string_buffer(root_dir_path.encode())
    success = lib.initialize(path, use_gpu, cpu_num_threads, load_all_models)
    if not success:
        raise Exception(lib.last_error_message().decode())

def load_model(speaker_id: int):
    success = lib.load_model(speaker_id)
    if not success:
        raise Exception(lib.last_error_message().decode())

def is_model_loaded(speaker_id: int) -> bool:
    return lib.is_model_loaded(speaker_id)

def metas() -> str:
    return lib.metas().decode()


def supported_devices() -> str:
    return lib.supported_devices().decode()


def variance_forward(length: int, phonemes: array, accents: array, speaker_id: array) -> Tuple[array, array]:
    pitches = numpy.zeros((length, ), dtype=numpy.float32)
    durations = numpy.zeros((length, ), dtype=numpy.float32)
    success = lib.variance_forward(length, phonemes, accents, speaker_id, pitches, durations)
    if not success:
        raise Exception(lib.last_error_message().decode())
    return pitches, durations


def decode_forward(length: int, phonemes: array, pitches: array, durations: array, speaker_id: array) -> array:
    wave_size = 0
    for i in range(length):
        wave_size += int(durations[i] * default_sampling_rate)
    output = numpy.zeros((wave_size,), dtype=numpy.float32)
    success = lib.decode_forward(
        length, phonemes, pitches, durations, speaker_id, output
    )
    if not success:
        raise Exception(lib.last_error_message().decode())
    return output

def sharevox_load_openjtalk_dict(dict_path: str):
    errno = lib.sharevox_load_openjtalk_dict(dict_path.encode())
    if errno != 0:
        raise Exception(lib.sharevox_error_result_to_message(errno).decode())

def sharevox_tts(text: str, speaker_id: int) -> bytes:
    output_binary_size = c_int()
    output_wav = POINTER(c_uint8)()
    errno = lib.sharevox_tts(text.encode(), speaker_id, byref(output_binary_size), byref(output_wav))
    if errno != 0:
        raise Exception(lib.sharevox_error_result_to_message(errno).decode())
    output = create_string_buffer(output_binary_size.value * sizeof(c_uint8))
    memmove(output, output_wav, output_binary_size.value * sizeof(c_uint8))
    lib.sharevox_wav_free(output_wav)
    return output

def sharevox_tts_from_kana(text: str, speaker_id: int) -> bytes:
    output_binary_size = c_int()
    output_wav = POINTER(c_uint8)()
    errno = lib.sharevox_tts_from_kana(text.encode(), speaker_id, byref(output_binary_size), byref(output_wav))
    if errno != 0:
        raise Exception(lib.sharevox_error_result_to_message(errno).decode())
    output = create_string_buffer(output_binary_size.value * sizeof(c_uint8))
    memmove(output, output_wav, output_binary_size.value * sizeof(c_uint8))
    lib.sharevox_wav_free(output_wav)
    return output

def finalize():
    lib.finalize()
