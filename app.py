
import torch
import librosa
import gradio as gr
from efficientnet_pytorch import EfficientNet
import torch.nn as nn

# ensemble_to_instruments 사전 및 고유한 악기 목록 생성
ensemble_to_instruments = {
    "Piano Quintet": ["Piano", "Violin", "Viola", "Cello"],
    "Solo Piano": ["Piano"],
    "Solo Cello" : ["Cello"],
    "Piano Trio": ["Piano", "Violin", "Cello"],
    "Viola Quintet": ["Violin", "Viola", "Cello"],
    "String Quartet": ["Violin", "Viola", "Cello"],
    "Clarinet Quintet": ["Clarinet", "Violin", "Viola", "Cello"],
    "Pairs Clarinet-Horn-Bassoon": ["Clarinet", "Horn", "Bassoon"],
    "Wind Quintet": ["Flute", "Oboe", "Clarinet", "Bassoon", "Horn"],
    "Accompanied Cello": ["Cello", "Piano"],
    "Accompanied Clarinet": ["Clarinet", "Piano"],
    "Wind and Strings Octet": ["Flute", "Oboe", "Clarinet", "Bassoon", "Violin", "Viola", "Cello"],
    "String Sextet": ["Violin", "Viola", "Cello"],
    "Piano Quartet": ["Piano", "Violin", "Viola", "Cello"],
    "Horn Piano Trio": ["Horn", "Piano", "Violin"],
    "Solo Violin": ["Violin"],
    "Solo Flute": ["Flute"],
    "Violin and Harpsichord": ["Violin", "Harpsichord"],
    "Clarinet-Cello-Piano Trio": ["Clarinet", "Cello", "Piano"],
    "Accompanied Violin": ["Violin", "Piano"],
    "Wind Octet": ["Oboe", "Clarinet", "Bassoon", "Horn"]
}

unique_instruments = set()
for instruments in ensemble_to_instruments.values():
    unique_instruments.update(instruments)

instrument_dict = {instrument: idx for idx, instrument in enumerate(unique_instruments)}

# EfficientNet 모델 정의 및 수정
class EfficientNetClassifier(nn.Module):
    def __init__(self, num_classes, in_channels=1):
        super(EfficientNetClassifier, self).__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b0')
        self.model._conv_stem = nn.Conv2d(in_channels, self.model._conv_stem.out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.model._conv_stem.weight.data = self.model._conv_stem.weight.data.mean(dim=1, keepdim=True).repeat(1, in_channels, 1, 1) / in_channels
        self.model._fc = nn.Linear(self.model._fc.in_features, num_classes)
    
    def forward(self, x):
        x = self.model(x)
        return x

num_classes = len(unique_instruments)
model = EfficientNetClassifier(num_classes=num_classes, in_channels=1)
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

def classify_audio(file):
    audio, sr = librosa.load(file, sr=16000)
    melspec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    melspec_db = librosa.power_to_db(melspec, ref=np.max)
    input_tensor = torch.tensor(melspec_db, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        predicted = (torch.sigmoid(output) > 0.5).float().squeeze().tolist()
    instrument_labels = [k for k, v in instrument_dict.items() if predicted[v] == 1]
    return ', '.join(instrument_labels)

iface = gr.Interface(fn=classify_audio, 
                     inputs=gr.Audio(source="upload", type="filepath"), 
                     outputs="text",
                     live=True,
                     title="Music Instrument Recognition",
                     description="Upload an audio file to classify the musical instruments.")

iface.launch()
