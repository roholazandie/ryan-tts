## Installation

```
mkdir -p outputs/phoneme_files
mkdir outputs/wav_files
```

```
pip install -r requirements.txt
```
and install the latest torch

download the models and change the config.json to point to them.

Note:
for now the tts models should be on the root of project, just like what we have in config.json

## Running
Make sure you change the paths and port in run.sh
```
./run.sh
```



## Using
To call it or ask a question:
```
curl --header "Content-Type: application/json" --request POST --data '{"input_text":"This is an awesome example."}' http://localhost:6666/api/tts
```

To download the wav files:
```
curl -o output.wav --header "Content-Type: application/json" --request POST --data '{"filename":"a1b8f97d-97c5-40c8-acd4-cbadca67abdf"}' http://localhost:3333/api/download
```
To delete:
```
curl --header "Content-Type: application/json" --request POST --data '{"filename":"a1b8f97d-97c5-40c8-acd4-cbadca67abdf"}' http://localhost:3333/api/delete
```
and delete all:
```
curl --header "Content-Type: application/json" http://localhost:3333/api/delete_all
```

