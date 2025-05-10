# Python Seq2Seq TomAI

A standalone Python chatbot built from scratch using a GRU-based sequence-to-sequence model.  
Trains on both the Cornell Movie-Dialogs Corpus and the Hugging Face DailyDialog dataset, with support for:

- **Training** (`--mode train`): mini-batch GRU training with teacher forcing, checkpointing, and millisecond timing.  
- **Chatting** (`--mode chat`): interactive shell interface with simple one-turn context memory.  
- **Save / Load**: models are saved to `chatbot.pth` (and periodic checkpoints `cp_<iter>.pth`).

---

## Features

- Encoder–Decoder architecture (GRU) implemented in PyTorch  
- Mini-batch training (configurable batch size) for speed  
- Automatic download & extraction of Cornell Movie dialogs  
- Integration of DailyDialog from Hugging Face 🤗  
- Command-line flags for hyperparameters (iterations, learning rate, batch size, print interval)  
- Graceful handling of missing data or model files  

---

## Requirements

- Python **3.7+**  
- [PyTorch](https://pytorch.org/) (CPU or GPU build)  
- [datasets](https://github.com/huggingface/datasets) (Hugging Face Datasets)  
- Standard libraries: `argparse`, `urllib`, `zipfile`, `shutil`, `re`, `random`, `time`

---

## Installation
Run the following in a terminal:


git clone https://github.com/perbra123/TomAI.git
cd TomAI
pip install torch datasets



Usage
The main script is ai.py. It supports two modes:


1. Train the bot:
   
--mode train  : enter training mode

--data_dir  : directory to download/extract Cornell data (default: data)

--iters   : total training iterations (one iteration = one mini-batch)

--pe     : print/save interval in iterations

--batch-size: number of examples per mini-batch

--lr     : learning rate for Adam optimizer

Checkpoints are saved as cp_<iter>.pth every pe iterations, and the final model as chatbot.pth.

Example usage:

python3 ai.py --mode train --iters 8000 --pe 500 
(runs in training mode, at 8000 total iters with default batch size (512) with checkpoint every 500 iters.)

2. Chat with the trained bot
Load the saved model and enter an interactive shell:

python ai.py --mode chat
Type your messages at the You: prompt


Enter quit to exit
