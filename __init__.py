import sys
import os
now_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(now_dir, 'third_party/Matcha-TTS'))
sys.path.append(now_dir)
model_dir = os.path.join(now_dir, "pretrained_models")
if not os.path.exists(os.path.join(model_dir,"CosyVoice2-0.5B")):
    print("download.......CosyVoice")
    from modelscope import snapshot_download
    snapshot_download('iic/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice2-0.5B')
    snapshot_download('iic/CosyVoice-ttsfrd', local_dir='pretrained_models/CosyVoice-ttsfrd')
    os.system(f'cd {model_dir}/CosyVoice-ttsfrd/ && pip install ttsfrd_dependency-0.1-py3-none-any.whl && pip install ttsfrd-0.4.2-cp310-cp310-linux_x86_64.whl && apt install -y unzip && unzip resource.zip -d .')

from cosyvoice.cli.cosyvoice import  CosyVoice2
from cosyvoice.utils.common import set_all_random_seed
import torchaudio
import torch
import librosa

max_val = 0.8
prompt_sr, target_sr = 16000, 24000

def postprocess(audio, top_db=60, hop_length=220, win_length=440):
    waveform = audio['waveform'].squeeze(0)
    source_sr = audio['sample_rate']
    speech = waveform.mean(dim=0,keepdim=True)
    if source_sr != prompt_sr:
        speech = torchaudio.transforms.Resample(orig_freq=source_sr, new_freq=prompt_sr)(speech)
    speech, _ = librosa.effects.trim(
        speech, top_db=top_db,
        frame_length=win_length,
        hop_length=hop_length
    )
    if speech.abs().max() > max_val:
        speech = speech / speech.abs().max() * max_val
    speech = torch.concat([speech, torch.zeros(1, int(target_sr * 0.2))], dim=1)
    return speech


class CosyVoice():
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO",),
                "text": ("TEXT",),
                "model": (['3s复刻', '跨语种复刻', '语言控制'],{
                    "default": "3s复刻"
                }),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 1.5, "step": 0.1}),
                "seed":("INT",{
                    "default": 8989
                }),
            },
            "optional":{
                "prompt": ("TEXT", ),
                "instuct": ("TEXT", ),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "run"
    CATEGORY = "GKK·CosVoice"
    _cosyvoice = CosyVoice2(os.path.join(model_dir,"CosyVoice2-0.5B"), load_jit=True, load_onnx=False, load_trt=False)

    def models(self):
        return os.listdir(model_dir)
    
    def reload(self,model="CosyVoice2-0.5B",load_jit=True, load_onnx=False, load_trt=False):
        self._cosyvoice = CosyVoice2(os.path.join(model_dir,model), load_jit=load_jit, load_onnx=load_onnx, load_trt=load_trt)

    def run(self,audio, text, model, speed, seed,  prompt=None, instuct=None):
        set_all_random_seed(seed)
        prompt_speech_16k = postprocess(audio)
        speechs = []
        if model == "跨语种复刻":
            speechs = [i["tts_speech"] for i in self._cosyvoice.inference_cross_lingual(text, prompt_speech_16k,  speed=speed)]
        elif model == "3s复刻":
            assert prompt is not None , '3s极速复刻 need prompt input'
            speechs = [i["tts_speech"] for i in self._cosyvoice.inference_zero_shot(text, prompt, prompt_speech_16k, speed=speed)]
        elif model == "语言控制":
            assert instuct is not None , '自然语言控制 need instuct input'
            speechs = [i["tts_speech"] for i in self._cosyvoice.inference_instruct2(text, instuct, prompt_speech_16k,  speed=speed)]
        tts_speech = torch.cat(speechs, dim=1)
        tts_speech = tts_speech.unsqueeze(0)
        return ({"waveform": tts_speech, "sample_rate": self._cosyvoice.sample_rate},)

class Copy3s(CosyVoice):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO",),
                "prompt": ("TEXT", ),
                "text": ("TEXT",),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 1.5, "step": 0.1}),
                "seed":("INT",{
                    "default": 8989
                }),
            },
        }
    def run(self,audio,prompt, text, speed, seed):
        return super().run(audio,text,"3s复刻",speed,seed,prompt)

class CrossLingual(CosyVoice):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO",),
                "text": ("TEXT",),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 1.5, "step": 0.1}),
                "seed":("INT",{
                    "default": 8989
                }),
            },
        }
    def run(self,audio, text, speed, seed):
        return super().run(audio,text,"跨语种复刻",speed,seed)

class NLControl(CosyVoice):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO",),
                "instuct": ("TEXT", ),
                "text": ("TEXT",),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 1.5, "step": 0.1}),
                "seed":("INT",{
                    "default": 8989
                }),
            },
        }
    def run(self,audio,instuct, text, speed, seed):
        return super().run(audio,text,"语言控制",speed,seed,instuct=instuct)
    
class TextNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"text": ("STRING", {"multiline": True, "dynamicPrompts": True})}}
    RETURN_TYPES = ("TEXT",)
    FUNCTION = "run"
    CATEGORY = "GKK·CosVoice"

    def run(self,text):
        return (text, )
    
NODE_CLASS_MAPPINGS = {
    "Text":TextNode,
    "CosyVoice3s":Copy3s,
    "CosyVoiceNLControl":NLControl,
    "CosyVoiceCrossLingual":CrossLingual,
    "CosyVoice":CosyVoice,
}

__all__ = ['NODE_CLASS_MAPPINGS']