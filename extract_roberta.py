import os
import soundfile
import signal
import torch
import numpy as np
import glob
import scipy.io as sio
from fairseq.models.wav2vec import Wav2VecModel
# def extract_wav2vec(wavfile, model_path):
#     print('Extracting wav2vec feature...')
#     cp = torch.load(model_path, map_location='cpu')
#     wav2vec = Wav2VecModel.build_model(cp['args'], task=None)
#     wav2vec.load_state_dict(cp['model'])
#     wav2vec.eval()
#     sample_rate = 16000
#     wavs, fs = soundfile.read(wavfile)
#     if fs != sample_rate:
#         result = int((wavs.shape[0]) / fs * sample_rate)
#         wavs = signal.resample(wavs, result)
#
#     if wavs.ndim > 1:
#         wavs = np.mean(wavs, axis=1)
#
#     wavs = torch.from_numpy(np.float32(wavs)).unsqueeze(0)
#
#     z = wav2vec.feature_extractor(wavs)
#     feature = wav2vec.feature_aggregator(z)
#     feature = feature.transpose(1, 2).squeeze(dim=0).detach().numpy()  # (t, 512)
#     return feature
def extract_wav2vec(wavfile, model_path):
    print('Extracting wav2vec feature...')
    cp = torch.load(model_path, map_location='cpu')
    wav2vec = Wav2VecModel.build_model(cp['args'], task=None)
    wav2vec.load_state_dict(cp['model'])
    wav2vec.eval()

    sample_rate = 16000
    wavs, fs = soundfile.read(wavfile)

    if fs != sample_rate:
        result = int((wavs.shape[0]) / fs * sample_rate)
        wavs = signal.resample(wavs, result)

    if wavs.ndim > 1:
        wavs = np.mean(wavs, axis=1)

    wavs = torch.from_numpy(np.float32(wavs)).unsqueeze(0)

    z = wav2vec.feature_extractor(wavs)
    feature = wav2vec.feature_aggregator(z)
    feature = feature.transpose(1, 2).squeeze(dim=0).detach().numpy()  # (t, 512)
    return feature
# 定义 extract_wav2vec 函数，和之前提供的一样
# 定义 IEMOCAP 数据集文件夹路径
data_folder = '/media/wp/data/dataset/IEMOCAP'

# 获取所有 WAV 文件的路径列表，遍历所有 Session 下的 WAV 文件
session_folders = glob.glob(os.path.join(data_folder, 'Session*', 'sentences', 'wav'))
wav_files = []
for session_folder in session_folders:
    session_wav_files = glob.glob(os.path.join(session_folder, '**', '*.wav'), recursive=True)
    wav_files.extend(session_wav_files)

# 加载 wav2vec 模型
wav2vec_model_path = '/media/wp/data/google-download/wav2vec_large.pt'
cp = torch.load(wav2vec_model_path, map_location='cpu')
wav2vec = Wav2VecModel.build_model(cp['args'], task=None)
wav2vec.load_state_dict(cp['model'])
wav2vec.eval()

# 定义填充后的长度
a_length = 460
from scipy import io
# 导入相关的工具函数
import utils
output_folder = '/media/wp/data/item/Key-Sparse-Transformer/wav_wav2vec_mat'
# 遍历所有 WAV 文件并提取特征并保存为 .mat 文件
for wavfile in wav_files:
    print(f"Processing: {wavfile}")
    x_a = extract_wav2vec(wavfile, wav2vec_model_path)

    # 进行填充
    x_a, x_a_padding_mask = utils.dataset.pad_input(x_a, a_length)
    x_a_np = x_a.numpy()
    # 获取 WAV 文件名，并去除扩展名（后缀名）
    wav_filename = os.path.splitext(os.path.basename(wavfile))[0]

    # 创建 .mat 文件名，以 WAV 文件名为基础
    mat_filename = os.path.join(output_folder, f"{wav_filename}.mat")

    # 创建一个字典，用于保存特征和掩码
    feature_dict = {"x_a": x_a_np, "x_a_padding_mask": x_a_padding_mask}
    # feature_dict = {"x_a": x_a}

    # 使用 scipy.io.savemat() 函数将特征和掩码保存为 .mat 文件
    sio.savemat(mat_filename, feature_dict)
    x0 = sio.loadmat(mat_filename)["x_a"]



