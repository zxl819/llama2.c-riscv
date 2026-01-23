import json
import os
import random
import torch
import numpy as np
from tokenizer import Tokenizer

# Configuration
DATA_FILE = "chat_dataset.json"
TOKENIZER_MODEL = "tokenizer.model"

class ChatDataset(torch.utils.data.IterableDataset):
    def __init__(self, split, max_seq_len, vocab_size, vocab_source):
        super().__init__()
        self.split = split
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.vocab_source = vocab_source
        
        # Load tokenizer
        self.tokenizer = Tokenizer(TOKENIZER_MODEL)
        
        # Load data
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            self.data = json.load(f)
            
        # Pre-tokenize data into a long sequence of tokens
        self.tokens = []
        for conversation in self.data:
            messages = conversation["messages"]
            # Simple formatting: [INST] User [/INST] Assistant
            # Note: This is a simplified version. Real Llama 2 chat templates are more complex.
            if len(messages) >= 2:
                user_msg = messages[0]["content"]
                asst_msg = messages[1]["content"]
                
                # Format: <s>[INST] user [/INST] assistant </s>
                # We rely on the tokenizer to add BOS/EOS if configured, but here we construct manually to be sure
                
                # [INST] and [/INST] are not special tokens in the base tokenizer usually, 
                # they are just text. Llama 2 tokenizer splits them.
                # Let's just format as string and encode.
                
                text = f"[INST] {user_msg} [/INST] {asst_msg}"
                
                # Encode with BOS=True, EOS=True
                t = self.tokenizer.encode(text, bos=True, eos=True)
                self.tokens.extend(t)
                
        self.tokens = np.array(self.tokens, dtype=np.uint16)
        print(f"Loaded {len(self.data)} conversations, total {len(self.tokens)} tokens.")

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        seed = 42 + (worker_info.id if worker_info else 0)
        rng = random.Random(seed)
        
        # Simple sliding window or random sampling
        # For training, we usually want random chunks
        data_len = len(self.tokens)
        while True:
            # Pick a random start position
            # Ensure we have enough tokens for max_seq_len + 1 (for x and y)
            if data_len <= self.max_seq_len + 1:
                # Data too short, just pad or loop? 
                # For this example, let's just yield the whole thing padded if needed, 
                # or just crash if too short (user needs more data).
                # Let's just loop from 0.
                ix = 0
            else:
                ix = rng.randint(0, data_len - self.max_seq_len - 1)
                
            chunk = torch.from_numpy((self.tokens[ix : ix + self.max_seq_len + 1]).astype(np.int64))
            x = chunk[:-1]
            y = chunk[1:]
            yield x, y

class Task:
    @staticmethod
    def iter_batches(batch_size, device, num_workers=0, **dataset_kwargs):
        ds = ChatDataset(**dataset_kwargs)
        dl = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, pin_memory=True, num_workers=num_workers
        )
        for x, y in dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            yield x, y
