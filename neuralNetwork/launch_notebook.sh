#!/bin/sh
if [ ! -d "venv" ]; then
    python -m pip install virtualenv
    echo creating virtual env
    python -m venv venv
    echo Created
    echo Installing Reqiured Packages...
    ./venv/bin/pip install -r requirements.txt
fi

./venv/bin/pip install -r requirements.txt -q
./venv/bin/jupyter notebook GalacticSearch-MobileNetV2.ipynb
