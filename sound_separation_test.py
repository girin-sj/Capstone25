import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
import csv
import os
import matplotlib.pyplot as plt
from scipy.io import wavfile
import scipy.signal
import tkinter as tk
from tkinter import filedialog, messagebox

# Load YAMNet model
model = hub.load('https://tfhub.dev/google/yamnet/1')

# Load class names
def class_names_from_csv(class_map_csv_text):
    class_names = []
    with tf.io.gfile.GFile(class_map_csv_text) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            class_names.append(row['display_name'])
    return class_names

class_map_path = model.class_map_path().numpy()
class_names = class_names_from_csv(class_map_path)

# Resample if needed
def ensure_sample_rate(original_sample_rate, waveform, desired_sample_rate=16000):
    if original_sample_rate != desired_sample_rate:
        desired_length = int(round(float(len(waveform)) / original_sample_rate * desired_sample_rate))
        waveform = scipy.signal.resample(waveform, desired_length)
    return desired_sample_rate, waveform

# 분석 함수
def analyze_audio(file_path):
    sample_rate, wav_data = wavfile.read(file_path)

    # 스테레오인 경우 모노로 변환
    if len(wav_data.shape) > 1 and wav_data.shape[1] == 2:
        wav_data = np.mean(wav_data, axis=1)

    # 샘플링 레이트 확인 및 리샘플링
    sample_rate, wav_data = ensure_sample_rate(sample_rate, wav_data)

    # 정규화
    waveform = wav_data / tf.int16.max

    # 모델 예측
    scores, embeddings, spectrogram = model(waveform)
    scores_np = scores.numpy()
    spectrogram_np = spectrogram.numpy()
    mean_scores = np.mean(scores_np, axis=0)

    # 결과 출력
    infered_class = class_names[mean_scores.argmax()]
    print(f"\n🟡 Detected main sound: {infered_class}\n")

    # 시각화
    plt.figure(figsize=(10, 6))
    plt.subplot(3, 1, 1)
    plt.plot(waveform)
    plt.title('Waveform')
    plt.xlim([0, len(waveform)])

    plt.subplot(3, 1, 2)
    plt.imshow(spectrogram_np.T, aspect='auto', interpolation='nearest', origin='lower')
    plt.title('Log-Mel Spectrogram')

    top_n = 10
    top_class_indices = np.argsort(mean_scores)[::-1][:top_n]
    plt.subplot(3, 1, 3)
    plt.imshow(scores_np[:, top_class_indices].T, aspect='auto', interpolation='nearest', cmap='gray_r')
    patch_padding = (0.025 / 2) / 0.01
    plt.xlim([-patch_padding-0.5, scores.shape[0] + patch_padding-0.5])
    plt.yticks(range(top_n), [class_names[i] for i in top_class_indices])
    plt.title('Top Class Scores Over Time')
    plt.tight_layout()
    plt.show()

    # 상위 5개 클래스
    print("🔝 Top 5 class predictions:")
    top_classes = tf.argsort(mean_scores, direction='DESCENDING')[:5]
    class_map_csv = hub.resolve("https://tfhub.dev/google/yamnet/1") + "/assets/yamnet_class_map.csv"
    class_map = pd.read_csv(class_map_csv, index_col='index')

    for tensor_idx in top_classes:
        idx = tensor_idx.numpy()
        try:
            class_name = class_map.loc[idx]['display_name']
            print(f"- {class_name} ({mean_scores[idx]:.2f})")
        except KeyError:
            print(f"- 알 수 없는 클래스 ID: {idx}")

# GUI 함수
def choose_file():
    file_path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
    if file_path:
        try:
            analyze_audio(file_path)
        except Exception as e:
            messagebox.showerror("오류", f"파일 분석 중 오류 발생: {e}")
    else:
        messagebox.showinfo("안내", "파일을 선택하지 않았습니다.")

# tkinter GUI 설정
root = tk.Tk()
root.title("오디오 파일 분석 (YAMNet)")
root.geometry("400x200")

label = tk.Label(root, text="분석할 .wav 파일을 선택하세요", font=("Arial", 14))
label.pack(pady=30)

button = tk.Button(root, text="📂 파일 선택", font=("Arial", 12), command=choose_file)
button.pack()

root.mainloop()
