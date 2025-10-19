#!/bin/bash

uv venv -p 3.12.4
source .venv/bin/activate
uv pip install -r requirements.txt
