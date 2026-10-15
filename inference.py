import os
import torch
from decord import VideoReader, cpu
import matplotlib.pyplot as plt
from holmesvau.holmesvau_utils import load_model, generate, show_smapled_video

mllm_path = './ckpts/HolmesVAU-2B'
sampler_path = './holmesvau/ATS/anomaly_scorer.pth'
device = torch.device('cuda:0')
model, tokenizer, generation_config, sampler = load_model(mllm_path, sampler_path, device)
'''
video_path = "./examples/robbery.mp4"
prompt = "Could you specify the anomaly events present in the video?"
pred, history, frame_indices, anomaly_score = generate(video_path, prompt, model, tokenizer, generation_config, sampler, select_frames=12, use_ATS=True)
print('\nUser:', prompt, '\nHolmesVAU:', pred)
'''
video_paths = ["./examples/robbery.mp4", "./examples/american_fire.mp4"]
prompt = "Could you specify the anomaly events present in the video?"
for video_path in video_paths:
    pred, history, frame_indices, anomaly_score = generate(
        video_path, prompt, model, tokenizer, generation_config, sampler, select_frames=12, use_ATS=True
    )
    print(video_path, pred)

