import torch
import torchaudio
import torch.nn.functional as F
import sys
import numpy as np

sys.path.append("rawnet")
import RawNet3 as RawNet3

class AudioInference:
    def __init__(self, model_path, device=None):
        """
        Инициализация инференса
        
        Args:
            model_path (str): Путь к сохраненным весам модели
            device (str, optional): Устройство для вычислений ('cuda' или 'cpu')
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path)
        self.labels = {
            0: 'angry',
            1: 'neutral', 
            2: 'other',
            3: 'positive',
            4: 'sad'
        }
        
    def _load_model(self, model_path):
        """Загрузка модели"""
        model = RawNet3.MainModel(
            nOut=256,
            encoder_type="ECA",
            sinc_stride=3,
            max_frame=200,
            sr=16000
        )
        model = model.to(self.device)
        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        return model
    
    def _preprocess_audio(self, audio_path):
        """
        Предобработка аудио файла
        
        Args:
            audio_path (str): Путь к аудио файлу
            
        Returns:
            torch.Tensor: Предобработанный аудио сигнал
        """
        signal, sample_rate = torchaudio.load(audio_path)
        
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            signal = resampler(signal)
        
        if signal.shape[0] != 1:
            signal = signal.mean(dim=0, keepdim=True)
            
        # 3 секунды
        target_size = 16000 * 3
        
        if signal.shape[1] < target_size:
            padding_size = target_size - signal.shape[1]
            signal = F.pad(signal, (0, padding_size))
        elif signal.shape[1] > target_size:
            start = (signal.shape[1] - target_size) // 2
            signal = signal[:, start:start + target_size]
        
        if len(signal.shape) == 2:
            signal = signal.unsqueeze(0)  # Добавляем размерность батча
            
        return signal.to(self.device)
    
    def predict_file(self, audio_path, return_probs=False):
        """
        Предсказание эмоции для аудио файла
        
        Args:
            audio_path (str): Путь к аудио файлу
            return_probs (bool): Вернуть ли вероятности для всех классов
            
        Returns:
            str: Предсказанная эмоция
            dict: Вероятности для всех классов (если return_probs=True)
        """
        signal = self._preprocess_audio(audio_path)
        
        with torch.no_grad():
            if len(signal.shape) != 3:
                raise ValueError(f"Неверная размерность входного тензора: {signal.shape}. Ожидается (batch_size, channel, time)")
                
            predictions = self.model(signal)
            probabilities = torch.exp(predictions)
            
            pred_idx = torch.argmax(probabilities).item()
            pred_label = self.labels[pred_idx]
            
            if return_probs:
                probs_dict = {self.labels[i]: prob.item() for i, prob in enumerate(probabilities[0])}
                return pred_label, probs_dict
            
            return pred_label
    
    def predict_batch(self, audio_paths):
        """
        Батчевое предсказание для нескольких файлов
        
        Args:
            audio_paths (list): Список путей к аудио файлам
            
        Returns:
            list: Список предсказанных эмоций
        """
        predictions = []
        for audio_path in audio_paths:
            pred = self.predict_file(audio_path)
            predictions.append(pred)
        return predictions

if __name__ == "__main__":
    model_path = "rawnet_model_91.pth"  # Укажите путь к вашей модели
    inferencer = AudioInference(model_path)
    
    audio_paths = [
        "c:\\Users\\Profi\\Downloads\\ru-uz\\data\\angry\\1f25ef18695e14efcbfa76e50081b455.wav",
        "c:\\Users\\Profi\\Downloads\\ru-uz\\data\\sad\\5d302e577468ca697d85d565c5834850.wav",
    ]
    batch_predictions = inferencer.predict_batch(audio_paths)
    for path, pred in zip(audio_paths, batch_predictions):
        print(f"File: {path}, Predicted emotion: {pred}")