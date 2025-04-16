import os
import sys
import torch
import librosa
import torchaudio
import gradio as gr
from funasr import AutoModel

from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
from cosyvoice.utils.common import set_all_random_seed

now_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(now_dir, 'third_party/Matcha-TTS'))
sys.path.append(now_dir)
model_dir = os.path.join(now_dir, "pretrained_models")

if not os.path.exists(os.path.join(model_dir, "CosyVoice2-0.5B")):
    print("download.......CosyVoice")
    from modelscope import snapshot_download
    snapshot_download('iic/CosyVoice2-0.5B',
                      local_dir='pretrained_models/CosyVoice2-0.5B')
    snapshot_download('iic/CosyVoice-ttsfrd',
                      local_dir='pretrained_models/CosyVoice-ttsfrd')
    os.system(f'cd {model_dir}/CosyVoice-ttsfrd/ && pip install ttsfrd_dependency-0.1-py3-none-any.whl && pip install ttsfrd-0.4.2-cp310-cp310-linux_x86_64.whl && apt install -y unzip && unzip resource.zip -d .')

max_val = 0.8
prompt_sr, target_sr = 16000, 24000


class CosVoice(CosyVoice2):
    _load = False
    _data = []

    def getData(self, index=None):
        if index is not None:
            yield self._data[index]
        else:
            for i in self._data:
                yield i

    def exec(self, *arg, **kwarg):
        yield from self._model(*arg, **kwarg)

    def getTTS(self, mode="自然语言控制", prompt_text=None, prompt_speech_16k=None, instruct_text=None, spk_id=""):
        self.prompt_speech_16k = prompt_speech_16k
        self.prompt_text = self.frontend.text_normalize(
            prompt_text, split=False) if prompt_text else None
        self.instruct_text = self.frontend.text_normalize(
            instruct_text, split=False) if instruct_text else None
        self.spk_id = spk_id
        self._load = True
        self._data = []
        self._model = None

        if mode == '预训练音色':
            assert self.spk_id, '没有可用的预训练音色！'
            self._model = self.sft
        elif mode == '3s极速复刻':
            assert  prompt_text ,'prompt文本为空'
            assert  prompt_speech_16k is not None, 'prompt音频为空'
            self._model = self.zero_shot
        elif mode == '跨语种复刻':
            assert  prompt_speech_16k is not None, 'prompt音频为空'
            self._model = self.cross_lingual
        else:
            assert instruct_text, 'instruct文本为空'
            self._model = self.instruct2
        return self._model

    def instructText(self, text=None):
        if text:
            return self.frontend.text_normalize(text, split=False)
        else:
            return self.instruct_text

    def sft(self, text, stream=False, speed=1.0, index=None):
        ps = self.frontend.frontend_sft(text, self.spk_id)
        yield from self.tts(ps, stream, speed, index)

    def zero_shot(self, text, stream=False, speed=1.0, index=None):
        ps = self.frontend.frontend_zero_shot(text, self.prompt_text, self.prompt_speech_16k, self.sample_rate)
        yield from self.tts(ps, stream, speed, index)

    def cross_lingual(self, text, stream=False, speed=1.0, index=None):
        ps = self.frontend.frontend_cross_lingual(text, self.prompt_speech_16k, self.sample_rate)
        yield from self.tts(ps, stream, speed, index)

    def instruct(self, text, stream=False, speed=1.0, instruct_text=None, index=None):
        ps = self.frontend.frontend_instruct(text, self.spk_id, self.instructText(instruct_text))
        yield from self.tts(ps, stream, speed, index)

    def instruct2(self, text, stream=False, speed=1.0, instruct_text=None, index=None):
        ps = self.frontend.frontend_instruct2(text,  self.instructText(instruct_text), self.prompt_speech_16k, self.sample_rate)
        yield from self.tts(ps, stream, speed, index)

    def tts(self, model_input, stream, speed, index=None):
        for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
            if index is not None:
                self._data[index] = model_output['tts_speech']
            else:
                self._data.append(model_output['tts_speech'])
            yield model_output['tts_speech']

    def split_text(self, text):
        self._data = []
        return "\n".join(self.frontend.text_normalize(text, split=True))

    def savePath(self, path):
        for i in self._data:
            torchaudio.save(os.path.join(path, 'sft_{}.wav'.format(i+300)),
                            i,
                            self.sample_rate)


def generate_seed():
    import random
    return {
        "__type__": "update",
        "value": random.randint(1, 100000000)
    }


def postprocess(speech, top_db=60, hop_length=220, win_length=440):
    speech, _ = librosa.effects.trim(
        speech,
        top_db=top_db,
        frame_length=win_length,
        hop_length=hop_length
    )
    if speech.abs().max() > max_val:
        speech = speech / speech.abs().max() * max_val
        speech = torch.concat([speech, torch.zeros(1, int(target_sr * 0.2))],
                              dim=1)
    return speech


def prompt_wav_recognition(prompt_wav,mode):
    if mode == '3s极速复刻':
        # "zn", "en", "yue", "ja", "ko", "nospeech"
        res = asr_model.generate(input=prompt_wav, language="auto", use_itn=True)
        return  res[0]["text"].split('|>')[-1]
    return ""


def generate(tts_text, mode_box, prompt_text, prompt_wav, instruct_text, seed, stream, speed):
    set_all_random_seed(seed)
    model = cosyvoice.getTTS(mode_box, prompt_text, postprocess(
        load_wav(prompt_wav, prompt_sr)), instruct_text)

    for text in tts_text.splitlines():
        print(text)
        for i in model(text, stream=stream, speed=speed):
            yield (target_sr, i.numpy().flatten())


def final():
    for i in cosyvoice.getData():
        yield (target_sr, i.numpy().flatten())
def single():
    for i, j in enumerate(cosyvoice.getData()):
        torchaudio.save('vc_{}.wav'.format(i), j, cosyvoice.sample_rate)

def main():
    instruct_dict = {'3s极速复刻': '1. 选择音频文件\n2. 修改prompt文本\n3. 输入合成文本\n4. 点击分割成句\n5. 检查合成句子\n6.点击生成音频',
                     '自然语言控制': '1. 选择音频文件\n2. 输入instruct文本\n3. 输入合成文本\n4. 点击分割成句\n5. 检查合成句子\n6.点击生成音频',
                     "跨语种复刻": "1. 选择音频文件\n2. 输入合成文本\n3. 点击分割成句\n4. 检查合成句子\n5.点击生成音频"
                    }
    inference_mode_list = [i for i in instruct_dict.keys()]

    with gr.Blocks() as demo:
        with gr.Row():
            mode_box = gr.Radio(choices=inference_mode_list,
                                label='选择模式',
                                value=inference_mode_list[0])
            instruction_text = gr.Text(
                label="操作步骤",
                value=instruct_dict[inference_mode_list[0]])
            stream = gr.Radio(
                choices=[('否', False), ('是', True)],
                label='流式推理',value=False,scale=0.5)
            speed = gr.Number(value=1, label="速度调节(仅支持非流式推理)", minimum=0.5, maximum=2.0, step=0.1)
            with gr.Column(scale=0.25):
                seed_button = gr.Button(value="\U0001F3B2")
                seed = gr.Number(value=36051420, label="随机推理种子")
        with gr.Row():
            prompt_wav_upload = gr.Audio(
                sources='upload', type='filepath', label='选择prompt音频文件，注意采样率不低于16khz，控制30s以内')
            with gr.Column():
                prompt_text = gr.Textbox(
                    label="prompt文本", placeholder="请输入prompt文本，支持自动识别，您可以自行修正识别结果...", value='')
                instruct_text = gr.Textbox(
                    label="输入instruct文本", placeholder="请输入instruct文本.例如:用四川话说这句话。", value='')
        tts_text = gr.Textbox(label="输入合成文本，注意换行(shift+emter)切分，单行不可过长", value="""CosyVoice迎来全面升级，提供更准、更稳、更快、 更好的语音生成能力。
CosyVoice is undergoing a comprehensive upgrade, providing more accurate, stable, faster, and better voice generation capabilities.""")

        with gr.Row():
            with gr.Column(scale=0.1):
                split_button = gr.Button("分割成句")
                generate_button = gr.Button("生成音频", visible=False)
                edit_button = gr.Button("修改单句音频", visible=False)
            audio_output = gr.Audio(label="结果", autoplay=True, streaming=True, visible=False)
        with gr.Row():
            final_btn = gr.Button("重新合成", visible=False, variant="primary")
            single_btn = gr.Button("单独保存", visible=False,)
        @gr.render(inputs=[tts_text,mode_box], triggers=[edit_button.click,])
        def split(tts_text, mode_box):
            with gr.Column():
                for i, text in enumerate(tts_text.splitlines()):
                    with gr.Row():
                        index_component = gr.Number(i, visible=False)
                        audio_component = gr.Audio((target_sr, next(cosyvoice.getData(i)).numpy().flatten()), scale=0.3, show_label=False)
                        with gr.Column():
                            texts = gr.Textbox(text, interactive=True,label="修改后点击->重新生成")
                            with gr.Row():
                                if mode_box == "自然语言控制":
                                    instruct_component = gr.Textbox(show_label=False, placeholder="请输入instruct文本, 默认："+instruct_text.value, value='')
                                    gr.Button("重新生成").click(
                                        lambda x, y, z: (target_sr, next(cosyvoice.exec(text=x, instruct_text=z, index=y)).numpy().flatten()),
                                        [texts, index_component,instruct_component],
                                        audio_component)
                                else:
                                    gr.Button("重新生成").click(
                                        lambda x, y: (target_sr, next(cosyvoice.exec(text=x, index=y)).numpy().flatten()),
                                        [texts, index_component],
                                        audio_component)
        seed_button.click(generate_seed, outputs=seed)
        mode_box.change(lambda x: instruct_dict[x],
                        mode_box,
                        instruction_text)
        prompt_wav_upload.change(prompt_wav_recognition,
                                 [prompt_wav_upload,mode_box],
                                 prompt_text)
        split_button.click(cosyvoice.split_text,
                           tts_text,
                           tts_text).then(
            lambda: (gr.update(visible=True),gr.update(visible=True)),
            outputs=[generate_button,audio_output])
        generate_button.click(generate,
                              [tts_text, mode_box, prompt_text,
                                  prompt_wav_upload,  instruct_text, seed, stream, speed],
                              audio_output).then(
            lambda: (gr.update(visible=False), gr.update(visible=True)),
            outputs=[split_button, edit_button])
        edit_button.click(lambda: (gr.update(visible=False),gr.update(visible=True), gr.update(visible=True),gr.update(visible=True)),
                          outputs=[edit_button,split_button, final_btn,single_btn])
        final_btn.click(final,
                        outputs=audio_output).then(
                            lambda: gr.update(visible=True),
                            outputs=split_button)
        single_btn.click(single)
    demo.launch(server_port=8000, server_name='0.0.0.0')


if __name__ == '__main__':
    cosyvoice = CosVoice('pretrained_models/CosyVoice2-0.5B')
    asr_model = AutoModel(
        model="iic/SenseVoiceSmall",
        disable_update=True,
        log_level='DEBUG',
        device="cuda:0")
    main()
