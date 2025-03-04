import os
import sys

import gc
import torch
import librosa
import torchaudio

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

from cosyvoice.cli.cosyvoice import  CosyVoice2,CosyVoice as CosyVoice1
from cosyvoice.utils.common import set_all_random_seed

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

class Loader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (os.listdir(model_dir),),
                "load_jit": ("BOOLEAN", {"default": True},),
                "load_onnx": ("BOOLEAN", {"default": False},),
                "load_trt": ("BOOLEAN", {"default": False},),
            },
        }
    RETURN_TYPES = ("MODEL_CosyVoice",)
    RETURN_NAMES = ("model",)
    FUNCTION = "run"
    CATEGORY = "GKK·CosVoice"
    def run(self, model, load_jit, load_onnx, load_trt):
        print("GKK·CosVoice: Model loading")
        if "300" in model:
            return (CosyVoice1(os.path.join(model_dir,model), load_jit=load_jit, load_onnx=load_onnx),)

        else:
            return (CosyVoice2(os.path.join(model_dir,model), load_jit=load_jit, load_onnx=load_onnx, load_trt=load_trt),)

class CosyVoice():
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL_CosyVoice",),
                "audio": ("AUDIO",),
                "text": ("TEXT",),
                "mode": (['3s复刻', '跨语种复刻', '语言控制'],{
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

    RETURN_TYPES = ("AUDIO","speechs_dict")
    RETURN_NAMES = ("audio","speechs")
    FUNCTION = "run"
    CATEGORY = "GKK·CosVoice"
    __model = None

    def run(self, model, audio, text, mode, speed, seed,  prompt=None, instuct=None, concat= False):
        set_all_random_seed(seed)
        prompt_speech_16k = postprocess(audio)
        speechs = []
        print("GKK·CosVoice: Start infer")
        if mode == "跨语种复刻":
            speechs = [i["tts_speech"] for i in model.inference_cross_lingual(text, prompt_speech_16k,  speed=speed)]
        elif mode == "3s复刻":
            assert prompt is not None , '3s极速复刻 need prompt input'
            speechs = [i["tts_speech"] for i in model.inference_zero_shot(text, prompt, prompt_speech_16k, speed=speed)]
        elif mode == "语言控制":
            assert instuct is not None , '自然语言控制 need instuct input'
            speechs = [i["tts_speech"] for i in model.inference_instruct2(text, instuct, prompt_speech_16k,  speed=speed)]
        gc.collect()
        torch.cuda.empty_cache()
        tts_speech = torch.cat(speechs, dim=1)
        tts_speech = tts_speech.unsqueeze(0)
        if concat:
            print("GKK·CosVoice: return speechs")
            return  ({"waveform": tts_speech, "sample_rate": model.sample_rate}, {"speechs":speechs, "sample_rate": model.sample_rate},)
        else:
            return ( {"waveform": tts_speech, "sample_rate": model.sample_rate},)

class Copy3s(CosyVoice):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL_CosyVoice",),
                "audio": ("AUDIO",),
                "prompt": ("TEXT", ),
                "text": ("TEXT",),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 1.5, "step": 0.1}),
                "seed":("INT",{
                    "default": 8989
                }),
            },
        }
    def run(self,model,audio,prompt, text, speed, seed, concat=False):
        return super().run(model,audio,text,"3s复刻",speed,seed,prompt,concat=concat)

class CrossLingual(CosyVoice):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL_CosyVoice",),
                "audio": ("AUDIO",),
                "text": ("TEXT",),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 1.5, "step": 0.1}),
                "seed":("INT",{
                    "default": 8989
                }),
            },
        }
    def run(self,model,audio, text, speed, seed):
        return super().run(model,audio,text,"跨语种复刻",speed,seed)

class NLControl(CosyVoice):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL_CosyVoice",),
                "audio": ("AUDIO",),
                "instuct": ("TEXT", ),
                "text": ("TEXT",),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 1.5, "step": 0.1}),
                "seed":("INT",{
                    "default": 8989
                }),
            },
        }
    def run(self,model,audio,instuct, text, speed, seed):
        return super().run(model,audio,text,"语言控制",speed,seed,instuct=instuct)
    
class Input:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "text": ("STRING",{ "dynamicPrompts": True,}),
                        "text2": ("STRING", {
                            "multiline": True,
                            "dynamicPrompts": True,
                            "style": "resize: vertical;",
                            "oninput": "this.style.height = 'auto'; this.style.height = (this.scrollHeight) + 'px';" 
                        })
                    }
                }
    RETURN_TYPES = ("TEXT","TEXT")
    FUNCTION = "run"
    CATEGORY = "GKK·CosVoice"
    def run(self,text,text2):
        return (text,text2)

class VoiceSonic(Copy3s):
    def run(self,model,audio,prompt, text, speed, seed):
        return super().run(model, audio, prompt, text,speed,seed,True)


NODE_CLASS_MAPPINGS = {
    "Text2":Input,
    "CosyVoiceLoader":Loader,
    "CosyVoice3s":Copy3s,
    "CosyVoiceNLControl":NLControl,
    "CosyVoiceCrossLingual":CrossLingual,
    "CosyVoiceSonic":VoiceSonic,
}

__all__ = ['NODE_CLASS_MAPPINGS']