# How to Run

## Requirements

- Python 3.12 (via pyenv)
- pip packages listed in `requirements.txt`

## Install dependencies

```bash
cd Desktop/Projects/Robotics
~/.pyenv/versions/3.12.6/bin/pip install -r requirements.txt
```

> First install takes a few minutes — downloads PyTorch and YOLOv8.

## Start the server

```bash
~/.pyenv/versions/3.12.6/bin/python web_app.py
```

Wait until you see:

```
All models loaded.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

## Open the app

Go to [http://localhost:8000](http://localhost:8000) in your browser.

## Use the demo

1. Click **Camera** to use your webcam, or **Video** to use the built-in sample video
2. All 4 algorithms run in real-time and update the panels
3. Use the **Perturbations** sliders at the bottom to add noise, blur, rotation, or brightness changes
4. Click **Stats** to open the detailed metrics panel

## Stop the server

Press `Ctrl+C` in the terminal.

## If port 8000 is already in use

```bash
lsof -ti :8000 | xargs kill -9
```
