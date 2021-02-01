#!/usr/bin/env bash
export FLASK_APP=tts_flask_main.py
export FLASK_ENV=development
export PYTHONPATH="${PYTHONPATH}:/home/rohola/codes/ryan-tts/espnet/"
flask run --port 6666