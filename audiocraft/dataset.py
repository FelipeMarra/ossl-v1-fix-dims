import torch
from torch.utils.data import Dataset
import torchvision
import pandas as pd
import numpy as np
from typing import Dict
import librosa


class SoundtrackDataset(Dataset):

    
    def __init__(
        self,
        path: str,
        train_or_valid
    ):
        self.path = path
        self.video_fps = 25
        self.max_video_frames = 30 * self.video_fps
        if train_or_valid == 'train':
            self.data = pd.read_csv("train_meta.csv")
        elif train_or_valid == 'valid':
            self.data = pd.read_csv("valid_meta.csv")
        else:
            print("train or valid? - error")
            exit()
        print(len(self.data))
    def __len__(self) -> int:
        return len(self.data)
    
    def normalize_audio(self, waveform, method='peak', eps=1e-10):

        waveform = np.array(waveform)
        
        if method == 'peak':
            peak = np.max(np.abs(waveform))
            return waveform / (peak + eps)
        
        elif method == 'rms':
            rms = np.sqrt(np.mean(waveform**2))
            return waveform / (rms + eps)
        
        elif method == 'minmax':
            max_val = np.max(waveform)
            min_val = np.min(waveform)
            return 2 * (waveform - min_val) / (max_val - min_val + eps) - 1
        
        elif method == 'standard':
            mean = np.mean(waveform)
            std = np.std(waveform)
            return (waveform - mean) / (std + eps)
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
    def load_audio(self, audio_path: str) -> torch.Tensor:
        waveform, sample_rate = librosa.load(audio_path, sr=32000)
        waveform = torch.Tensor(waveform)
        waveform = waveform.unsqueeze(0)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        

        if waveform.shape[1] > 30 * sample_rate:
            waveform = waveform[:, :int(30*sample_rate)]
        else:
            pad_length = (30 * sample_rate) - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_length))
        
        return self.normalize_audio(waveform)
    
    def load_video(self, video_path: str) -> torch.Tensor:
        video, _, info = torchvision.io.read_video(
            video_path, 
            pts_unit='sec',
            output_format='TCHW'
        )
        
        if info['video_fps'] != self.video_fps:
            original_length = video.shape[0]
            target_length = int(original_length * self.video_fps / info['video_fps'])
            video = torch.nn.functional.interpolate(
                video.permute(1, 0, 2, 3),
                size=target_length,
                mode='linear'
            ).permute(1, 0, 2, 3)
        
        target_frames = 30 * self.video_fps
        if video.shape[0] > target_frames:
            video = video[:target_frames]
        else:
            pad_frames = target_frames - video.shape[0]
            video = torch.nn.functional.pad(video, (0, 0, 0, 0, 0, 0, 0, pad_frames))
        
        return video

    def get_video_features(self, video_path):
        return torch.load(video_path)
    
    def generate_prompt(self, item):
        caption = item['caption']
        mood = item['mood']
        prompt = "a film soundtrack for a " + mood + " scene. " + caption
        return prompt

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        item = self.data.iloc[idx]
        
        prompt = self.generate_prompt(item)
        file_name = item['film_id'] + "_" + str(item['clip_id'])
    
        audio = self.load_audio(self.path + file_name + ".wav")
        
        
        video_features = self.get_video_features(self.path + file_name + ".pt")

        
        sample = {
            'prompt': prompt,
            'audio': audio,
            'video': video_features
        }
        
        return sample