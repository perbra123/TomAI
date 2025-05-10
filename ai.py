print("Loading libraries...")
import os
import re
import random
import argparse
import urllib.request
import zipfile
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import time
import datasets
from datasets import load_dataset  # <-- add this


# ------------------------
# Configuration
# ------------------------
DATA_URL = 'https://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip'

# ------------------------
# Data Preparation Utils
# ------------------------

def ensure_data(data_dir):
    lines_path = os.path.join(data_dir, 'movie_lines.txt')
    convs_path = os.path.join(data_dir, 'movie_conversations.txt')
    if os.path.exists(lines_path) and os.path.exists(convs_path):
        return
    print('Downloading corpus...')
    zip_path = 'cornell_data.zip'
    urllib.request.urlretrieve(DATA_URL, zip_path)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall('.')
    os.remove(zip_path)
    extracted_dir = None
    for entry in os.listdir('.'):
        if os.path.isdir(entry) and os.path.exists(os.path.join(entry, 'movie_lines.txt')):
            extracted_dir = entry
            break
    if not extracted_dir:
        raise RuntimeError('Could not locate extracted corpus folder')
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    os.rename(extracted_dir, data_dir)
    print('Data ready.')


def normalize_text(text):
    text = text.lower().strip()
    return re.sub(r"[^a-zA-Z0-9\s\?\.!']", '', text)

class Vocabulary:
    PAD_TOKEN = 'PAD'
    SOS_TOKEN = 'SOS'
    EOS_TOKEN = 'EOS'

    def __init__(self, min_count=1):
        self.min_count = min_count
        self.word2index = {self.PAD_TOKEN: 0, self.SOS_TOKEN: 1, self.EOS_TOKEN: 2}
        self.word2count = {}
        self.index2word = {0: self.PAD_TOKEN, 1: self.SOS_TOKEN, 2: self.EOS_TOKEN}
        self.num_words = 3

    def add_sentence(self, sentence):
        for w in sentence.split():
            self.word2count[w] = self.word2count.get(w, 0) + 1

    def build_vocab(self):
        for w, count in self.word2count.items():
            if count >= self.min_count:
                self.word2index[w] = self.num_words
                self.index2word[self.num_words] = w
                self.num_words += 1

# Load corpus pairs

def load_pairs(data_dir, max_len=10):
    id2line = {}
    with open(os.path.join(data_dir, 'movie_lines.txt'), encoding='iso-8859-1') as f:
        for l in f:
            parts = l.split(' +++$+++ ')
            if len(parts) == 5:
                id2line[parts[0]] = normalize_text(parts[4])
    pairs = []
    with open(os.path.join(data_dir, 'movie_conversations.txt'), encoding='iso-8859-1') as f:
        for l in f:
            ids = eval(l.split(' +++$+++ ')[3])
            for i in range(len(ids) - 1):
                a = id2line[ids[i]].split()
                b = id2line[ids[i+1]].split()
                if 1 < len(a) <= max_len and 1 < len(b) <= max_len:
                    pairs.append((' '.join(a), ' '.join(b)))
    return pairs
# Load DailyDialogue for training and stuff idk
def load_dailydialog_pairs(max_len=10):
    datasets.trust_remote_code=True
    dataset = load_dataset("daily_dialog")
    pairs = []
    for split in ['train', 'validation', 'test']:
        for dialog in dataset[split]:
            utterances = dialog['dialog']
            for i in range(len(utterances) - 1):
                a = normalize_text(utterances[i])
                b = normalize_text(utterances[i+1])
                a_words = a.split()
                b_words = b.split()
                if 1 < len(a_words) <= max_len and 1 < len(b_words) <= max_len:
                    pairs.append((' '.join(a_words), ' '.join(b_words)))
    return pairs

# Convert sentence to tensor

def tensor_from_sentence(vocab, sentence):
    idxs = [vocab.word2index.get(w, vocab.word2index[vocab.PAD_TOKEN]) for w in sentence.split()]
    idxs.append(vocab.word2index[vocab.EOS_TOKEN])
    return torch.tensor(idxs, dtype=torch.long)

# ------------------------
# Seq2Seq Models
# ------------------------

class EncoderRNN(nn.Module):
    def __init__(self, vs, es, hs, nl=1):
        super().__init__()
        self.embedding = nn.Embedding(vs, es)
        self.gru = nn.GRU(es, hs, nl, batch_first=True)

    def forward(self, x, h=None):
        return self.gru(self.embedding(x), h)

class DecoderRNN(nn.Module):
    def __init__(self, vs, es, hs, nl=1):
        super().__init__()
        self.embedding = nn.Embedding(vs, es)
        self.gru = nn.GRU(es, hs, nl, batch_first=True)
        self.out = nn.Linear(hs, vs)

    def forward(self, x, h):
        o, hn = self.gru(self.embedding(x), h)
        return self.out(o.squeeze(1)), hn

# ------------------------
# Training Function
# ------------------------
def train_bot(enc, dec, pairs, vocab, dev, iters=10000, pe=500, lr=1e-3):
    """
    Train the encoder and decoder models using mini-batches.
    """
    # Prepare dataset and loader
    from torch.utils.data import Dataset, DataLoader
    import torch.nn.utils.rnn as rnn_utils

    class ChatDataset(Dataset):
        def __init__(self, pairs, vocab):
            self.pairs = pairs
            self.vocab = vocab
        def __len__(self):
            return len(self.pairs)
        def __getitem__(self, idx):
            inp, tgt = self.pairs[idx]
            return tensor_from_sentence(self.vocab, inp), tensor_from_sentence(self.vocab, tgt)

    def collate_fn(batch):
        inputs, targets = zip(*batch)
        inputs_padded = rnn_utils.pad_sequence(inputs, batch_first=True,
            padding_value=vocab.word2index[vocab.PAD_TOKEN])
        targets_padded = rnn_utils.pad_sequence(targets, batch_first=True,
            padding_value=vocab.word2index[vocab.PAD_TOKEN])
        return inputs_padded, targets_padded

    dataset = ChatDataset(pairs, vocab)
    loader = DataLoader(dataset, batch_size=512, shuffle=True, num_workers=4, pin_memory=True, collate_fn=collate_fn)  # use multiple workers for faster data loading

    enc.train(); dec.train()
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.word2index[vocab.PAD_TOKEN])
    enc_opt = optim.Adam(enc.parameters(), lr=lr)
    dec_opt = optim.Adam(dec.parameters(), lr=lr)
    start = time.perf_counter()

    total_iters = 0
    for epoch in range(1, (iters // len(loader)) + 2):
        for batch_idx, (inp_batch, tgt_batch) in enumerate(loader, 1):
            total_iters += 1
            inp_batch = inp_batch.to(dev)
            tgt_batch = tgt_batch.to(dev)

            enc_opt.zero_grad(); dec_opt.zero_grad()
            _, hidden = enc(inp_batch)
            # Initialize decoder input batch of SOS tokens
            batch_size = inp_batch.size(0)
            dec_input = torch.full((batch_size,1),
                vocab.word2index[vocab.SOS_TOKEN], dtype=torch.long, device=dev)
            loss = 0

            # Decode each time step
            for t in range(tgt_batch.size(1)):
                dec_output, hidden = dec(dec_input, hidden)
                loss += criterion(dec_output, tgt_batch[:,t])
                teacher_force = torch.rand(batch_size, device=dev) < 0.5
                top1 = dec_output.argmax(dim=1)
                next_token = torch.where(teacher_force, tgt_batch[:,t], top1)
                dec_input = next_token.unsqueeze(1)

            loss.backward(); enc_opt.step(); dec_opt.step()

            if total_iters % pe == 0:
                avg_loss = loss.item() / tgt_batch.size(1)
                end = time.perf_counter()
                time_el = (end - start) * 1000
                print(f"Iter {total_iters}/{iters} Loss: {avg_loss:.4f} Time: {time_el:.1f} ms")
                start = time.perf_counter()
                torch.save({
                    'enc': enc.state_dict(),
                    'dec': dec.state_dict(),
                    'v2i': vocab.word2index
                }, f'cp_{total_iters}.pth')

            if total_iters >= iters:
                break
        if total_iters >= iters:
            break

    torch.save({
        'enc': enc.state_dict(),
        'dec': dec.state_dict(),
        'v2i': vocab.word2index
    }, 'chatbot.pth')
    print('Training complete. Model saved to chatbot.pth')

# ------------------------
# Chatting Function
# ------------------------

def run_bot(enc, dec, v2i, dev):
    enc.eval(); dec.eval()
    i2w = {i: w for w, i in v2i.items()}
    last_response = ""
    print("Type 'quit' to exit.")
    while True:
        user_input = input('You: ')
        if user_input.lower().startswith('quit'):
            print('Bye!')
            break
        context = normalize_text((last_response + ' ' + user_input).strip())
        inp = tensor_from_sentence(VOCAB, context).unsqueeze(0).to(dev)
        _, h = enc(inp)
        dec_input = torch.LongTensor([[v2i[Vocabulary.SOS_TOKEN]]]).to(dev)
        reply = []

        for _ in range(50):
            dec_output, h = dec(dec_input, h)
            idx = dec_output.argmax(1).item()
            if idx == v2i[Vocabulary.EOS_TOKEN]:
                break
            reply.append(i2w.get(idx, '<unk>'))
            dec_input = torch.LongTensor([[idx]]).to(dev)

        last_response = ' '.join(reply)
        print('Bot:', last_response)

# ------------------------
# Main
# ------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'chat'], default='chat')
    parser.add_argument('--data_dir', default='data')
    parser.add_argument('--iters', type=int, default=20000)
    parser.add_argument('--pe', type=int, default=500)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    global VOCAB
    # then in main, before training:

    if args.mode == 'train':
        ensure_data(args.data_dir)
        print('Loading Corpus Data...')
        movie_pairs = load_pairs(args.data_dir)
        print("Loading DailyDialog Data...")
        daily_pairs = load_dailydialog_pairs()
        pairs = movie_pairs + daily_pairs
        random.shuffle(pairs)

        print(f'Loaded {len(pairs)} sentence pairs.')
        VOCAB = Vocabulary(min_count=3)
        for inp, tgt in pairs:
            VOCAB.add_sentence(inp)
            VOCAB.add_sentence(tgt)
        VOCAB.build_vocab()
        print(f'Total vocabulary size: {VOCAB.num_words}')

        encoder = EncoderRNN(VOCAB.num_words, 256, 512, 2).to(device)
        decoder = DecoderRNN(VOCAB.num_words, 256, 512, 2).to(device)

        train_bot(encoder, decoder, pairs, VOCAB, device, iters=args.iters, pe=args.pe)

    else:
        try:
            checkpoint = torch.load('chatbot.pth', map_location=device)
        except FileNotFoundError:
            print('No trained model found. Please run with --mode train first.')
            return

        v2i = checkpoint['v2i']
        VOCAB = Vocabulary()
        VOCAB.word2index = v2i
        VOCAB.index2word = {i: w for w, i in v2i.items()}
        VOCAB.num_words = len(v2i)

        encoder = EncoderRNN(VOCAB.num_words, 256, 512, 2).to(device)
        decoder = DecoderRNN(VOCAB.num_words, 256, 512, 2).to(device)
        encoder.load_state_dict(checkpoint['enc'])
        decoder.load_state_dict(checkpoint['dec'])
        run_bot(encoder, decoder, v2i, device)

if __name__ == '__main__':
    main()
