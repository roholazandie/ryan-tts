import time
import torch
from espnet2.bin.tts_inference import Text2Speech
from g2p_en import G2p
from parallel_wavegan.utils import download_pretrained_model
from parallel_wavegan.utils import load_model
import os, re
import numpy as np
from scipy.io.wavfile import write
from tacotron_cleaner.cleaners import custom_english_cleaners
import yaml
from config import TTSConfig

config = TTSConfig.from_json_file("config.json")

# train_config = "exp/tts_train_raw_phn_tacotron_g2p_en_no_space/config.yaml"
# model_file = "exp/tts_train_raw_phn_tacotron_g2p_en_no_space/200epoch.pth"

# train_config = "exp/tts_train_fastspeech_raw_phn_tacotron_g2p_en_no_space/config.yaml"
# model_file = "exp/tts_train_fastspeech_raw_phn_tacotron_g2p_en_no_space/1000epoch.pth"

# train_config = "exp/tts_train_fastspeech2_raw_phn_tacotron_g2p_en_no_space/config.yaml"
# model_file = "exp/tts_train_fastspeech2_raw_phn_tacotron_g2p_en_no_space/1000epoch.pth"

# train_config = "exp/tts_train_conformer_fastspeech2_raw_phn_tacotron_g2p_en_no_space/config.yaml"
# model_file = "exp/tts_train_conformer_fastspeech2_raw_phn_tacotron_g2p_en_no_space/1000epoch.pth"


# with open(esp_config.dict_path) as f:
#     lines = f.readlines()
# lines = [line.replace("\n", "").split(" ") for line in lines]
# char_to_id = {c: int(i) for c, i in lines}
# g2p = G2p()


cmu_phonemes = ["F", "M", "N", "L", "D", "B", "HH", "P", "T", "S", "R", "AE", "W", "Z", "V", "G", "NG", "DH", "AX",
                "AA", "AH", "AO", "AW", "AXR", "AY", "CH", "EH", "ER", "EY", "IH", "IX", "IY", "JH", "OW", "OY", "SH",
                "TH", "UH", "UW", "Y", "TS", "R", "R", "AH", "AA", "SIL", "IY", "L", "L", "R", "IH", ]


def extract_phonemes(text2speech, text, cmu_syle=True):
    phonemes = text2speech.preprocess_fn.tokenizer.text2tokens(text)

    if cmu_syle:
        cleaned_phonemes = []
        for phone in phonemes:
            cleaned_phone = re.sub(r'\d+', '', phone)
            cleaned_phonemes.append(cleaned_phone)
        return cleaned_phonemes
    return phonemes


train_config_dict = yaml.load(open(config.train_config))
tts_model = train_config_dict['tts']


def regulate_phoneme_duration(phoneme, start, end):
    for char in ['0', '1', '2', '3']:
        if char in phoneme:
            phoneme = phoneme.replace(char, '')
    if phoneme not in cmu_phonemes:
        phoneme = "SIL"

    start = int(float(start) / 10) + 10
    end = int(float(end) / 10) + 10
    return phoneme, start, end


tts_model = "tactron"

if tts_model == "tactron" or tts_model == "tactron2":
    text2speech = Text2Speech(
        train_config=config.train_config,
        model_file=config.model_file,
        device="cuda",
        # Only for Tacotron 2
        threshold=0.5,
        minlenratio=0.0,
        maxlenratio=10.0,
        use_att_constraint=False,
        backward_window=1,
        forward_window=3
    )
elif tts_model == "fastspeech" or tts_model == "fastspeech2":
    text2speech = Text2Speech(
        train_config=config.train_config,
        model_file=config.model_file,
        device="cuda",
        # Only for FastSpeech & FastSpeech2
        speed_control_alpha=1.0,
    )
else:
    raise Exception("Unknown tts_model")

text2speech.spc2wav = None  # Disable griffin-lim
# NOTE: Sometimes download is failed due to "Permission denied". That is
#   the limitation of google drive. Please retry after serveral hours.

# vocoder_model = "exp/train_nodev_ryanspeech_parallel_wavegan.v1/checkpoint-400000steps.pkl"
# vocoder_model = "/media/rohola/data/speech/ryan_speech_models/processes_model/exp/train_nodev_ryanspeech_parallel_wavegan.v1/checkpoint-400000steps.pkl"
vocoder = load_model(config.vocoder_model).to("cuda").eval()
vocoder.remove_weight_norm()

print(f"Input your favorite sentence in.")
text = input()


# synthesis
with torch.no_grad():
    start = time.time()
    wav, outs, outs_denorm, probs, att_ws, durations, focus_rate = text2speech(text)
    wav = vocoder.inference(outs)
    # extract phonemes
    phonemes = extract_phonemes(text2speech, text)

y = wav.view(-1).cpu().tolist()
durations = durations.tolist()
durations = durations[1:]
rtf = (time.time() - start) / (len(wav) / config.fs)
print(f"RTF = {rtf:5f}")

####################################
audio_duration = (len(y) / config.fs) * 1000

unit_duration = audio_duration / sum(durations)

ends = np.cumsum(durations) * unit_duration
starts = [0] + ends[:-1].tolist()

lines = []
phoneme_out = {"phonemes": [], "start": [], "end": []}
# phonemes_file = os.path.join(esp_config.phonemes_dir, out_file_name+".txt")
phonemes_file = os.path.join(config.phonemes_dir, "out.txt")
with open(phonemes_file, 'w') as file_writer:
    for phoneme, start, end in zip(phonemes, starts, ends):
        phoneme, start, end = regulate_phoneme_duration(phoneme, start, end)
        line = "{:4d} 0    0    0    0  {:4d} {:4s} 0.0000 ".format(start, end, phoneme) + '\n'
        file_writer.write(line)
        lines.append(line)
        phoneme_out["phonemes"].append(phoneme)
        phoneme_out["start"].append(start)
        phoneme_out["end"].append(end)

# let us listen to generated samples
write("fourth.wav", config.fs, wav.view(-1).cpu().numpy())
