import json


class TTSConfig:

    def __init__(self,
                 train_config="",
                 model_file="",
                 vocoder_model="",
                 phonemes_dir="",
                 voice_dir="",
                 fs="",
                 port="",
                 device="cuda"
                 ):

        self.train_config = train_config
        self.model_file = model_file
        self.vocoder_model = vocoder_model
        self.phonemes_dir = phonemes_dir
        self.voice_dir = voice_dir
        self.fs = fs
        self.port = port
        self.device = device

    @classmethod
    def from_dict(cls, json_object):
        config = TTSConfig()
        for key in json_object:
            config.__dict__[key] = json_object[key]
        return config

    @classmethod
    def from_json_file(cls, json_file):
        with open(json_file) as f:
            config_json = f.read()

        return cls.from_dict(json.loads(config_json))

    def __str__(self):
        return str(self.__dict__)