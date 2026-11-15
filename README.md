# MuseLSTM — AI Creative Text Generator

A character-level LSTM text-generation project built with Python, PyTorch and Streamlit.

## Features
- Train a two-layer LSTM on a custom text corpus.
- Generate text from a user-provided starting phrase.
- Adjust temperature to control randomness.
- Set the maximum number of newly generated characters.
- Download generated text from the Streamlit interface.

## 1. Requirements
- Python 3.10 or newer recommended
- pip
- Internet connection for installing packages

## 2. Create and activate a virtual environment (Windows PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\activate
```

## 3. Install dependencies

```powershell
pip install -r requirements.txt
```

If PyTorch installation fails, use the official selector at https://pytorch.org/get-started/locally/ to choose the correct installation command for your system.

## 4. Prepare the corpus

Edit `dataset/corpus.txt`. Add a reasonably large collection of text. The starter corpus is only for testing the complete pipeline; a larger corpus generally improves output quality. Use text you own or have permission to use.

## 5. Train

From the MuseLSTM project root:

```powershell
python train.py
```

The script saves:
- `models/lstm_model.pth`
- `models/vocab.json`

Training runs on CUDA when available; otherwise it uses CPU. CPU training may take longer.

## 6. Generate text in the terminal

```powershell
python generate.py
```

## 7. Launch the Streamlit app

```powershell
streamlit run app.py
```

Open the local URL printed by Streamlit, usually http://localhost:8501.

## How temperature works
- Lower temperature: more predictable character choices.
- Higher temperature: more varied and random character choices.

## Project structure

```text
MuseLSTM/
├── dataset/
│   └── corpus.txt
├── models/
│   └── (generated after training)
├── train.py
├── generate.py
├── app.py
├── requirements.txt
└── README.md
```

## Important note
This is a character-level language model. It learns patterns from the supplied corpus; it does not understand meaning like a modern large language model. The quality of generated text depends heavily on corpus size and quality.
