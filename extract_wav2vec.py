import os
import torch
import glob
import scipy.io as sio
from fairseq.models.roberta import RobertaModel

def extract_roberta(txtfile, model_path):
    print('Extracting RoBERTa feature...')
    roberta = RobertaModel.from_pretrained(model_path, checkpoint_file='model.pt')
    roberta.eval()

    with open(txtfile, 'r') as f:
        text = f.read()
    tokens = roberta.encode(text)
    embedding = roberta.extract_features(tokens)

    embedding = embedding.squeeze(dim=0).detach().numpy()   # (t, 1024)
    return embedding

roberta_model_path = '/media/wp/data/google-download/roberta.large'  # 替换为实际的 RoBERTa 模型路径

import glob

# Get all the .txt files in the '/media/wp/data/item/Key-Sparse-Transformer/raw_text' folder
text_files = glob.glob(os.path.join('/media/wp/data/item/Key-Sparse-Transformer/raw_text', '*.txt'))

import os
import scipy.io as sio
import utils
output_folder = '/media/wp/data/item/Key-Sparse-Transformer/text_roberta_mat'  # 替换为实际的文件夹路径
t_length = 20
for text_file in text_files:
    print(f"Processing: {text_file}")
    x_t = extract_roberta(text_file, roberta_model_path)
    x_t, x_t_padding_mask = utils.dataset.pad_input(x_t, t_length)
    x_t_np = x_t.numpy()
    # 获取文本文件名，并去除扩展名（后缀名）
    text_filename = os.path.splitext(os.path.basename(text_file))[0]

    # 创建 .mat 文件名，以文本文件名为基础，并指定保存路径
    mat_filename = os.path.join(output_folder, f"{text_filename}.mat")

    # 创建一个字典，用于保存特征
    feature_dict = {"x_t": x_t_np, "x_t_padding_mask": x_t_padding_mask}

    # 使用 scipy.io.savemat() 函数将特征保存为 .mat 文件
    sio.savemat(mat_filename, feature_dict)
