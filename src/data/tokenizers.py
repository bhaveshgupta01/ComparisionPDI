import re
import pandas as pd
import json
import os

class SMILESTokenizer:
    def __init__(self, vocab_file=None, max_len=100):
        self.max_len = max_len
        self.pad_token = '<pad>'
        self.cls_token = '<cls>'
        self.sep_token = '<sep>'
        self.unk_token = '<unk>'
        self.special_tokens = [self.pad_token, self.cls_token, self.sep_token, self.unk_token]
        self.vocab = {}
        self.inverse_vocab = {}
        # Regex based on Technical Specification 6.4.1
        self.regex = re.compile(r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])")
        
        if vocab_file and os.path.exists(vocab_file):
            self.load_vocab(vocab_file)
        else:
            self.build_initial_vocab()
            
    def build_initial_vocab(self):
        self.vocab = {tok: i for i, tok in enumerate(self.special_tokens)}
        self.inverse_vocab = {i: tok for tok, i in self.vocab.items()}
        
    def build_vocab(self, smiles_list):
        self.build_initial_vocab()
        unique_tokens = set()
        for smiles in smiles_list:
            if isinstance(smiles, str):
                tokens = self.tokenize(smiles)
                unique_tokens.update(tokens)
            
        for tok in sorted(list(unique_tokens)):
            if tok not in self.vocab:
                idx = len(self.vocab)
                self.vocab[tok] = idx
                self.inverse_vocab[idx] = tok
                
    def tokenize(self, smiles):
        return [t for t in self.regex.findall(smiles)]
        
    def encode(self, smiles):
        """
        Encode a SMILES string into a list of token IDs.
        """
        tokens = self.tokenize(smiles)
        # truncate
        tokens = tokens[:self.max_len - 2] # leave room for cls and sep
        
        token_ids = [self.vocab[self.cls_token]]
        for tok in tokens:
            token_ids.append(self.vocab.get(tok, self.vocab[self.unk_token]))
        token_ids.append(self.vocab[self.sep_token])
        
        # pad
        padding_length = self.max_len - len(token_ids)
        if padding_length > 0:
            token_ids.extend([self.vocab[self.pad_token]] * padding_length)
            
        return token_ids
        
    def save_vocab(self, vocab_file):
        with open(vocab_file, 'w') as f:
            json.dump(self.vocab, f)
            
    def load_vocab(self, vocab_file):
        with open(vocab_file, 'r') as f:
            self.vocab = json.load(f)
            self.inverse_vocab = {int(v): k for k, v in self.vocab.items()}


class ProteinTokenizer:
    def __init__(self, max_len=1200):
        self.max_len = max_len
        self.pad_token = '<pad>'
        self.cls_token = '<cls>'
        self.sep_token = '<sep>'
        
        # 20 standard AAs + X based on Technical Specification 6.4.2
        self.amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 
                            'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'X']
        self.special_tokens = [self.pad_token, self.cls_token, self.sep_token]
        
        self.vocab = {tok: i for i, tok in enumerate(self.special_tokens + self.amino_acids)}
        self.inverse_vocab = {i: tok for tok, i in self.vocab.items()}
        
    def tokenize(self, sequence):
        return list(sequence.upper())
        
    def encode(self, sequence):
        tokens = self.tokenize(sequence)
        tokens = [t if t in self.vocab else 'X' for t in tokens]
        # truncate
        tokens = tokens[:self.max_len - 2]
        
        token_ids = [self.vocab[self.cls_token]]
        for tok in tokens:
            token_ids.append(self.vocab[tok])
        token_ids.append(self.vocab[self.sep_token])
        
        padding_length = self.max_len - len(token_ids)
        if padding_length > 0:
            token_ids.extend([self.vocab[self.pad_token]] * padding_length)
            
        return token_ids

def build_and_save_smiles_vocab(train_csv_path, vocab_save_path):
    print(f"Reading SMILES from {train_csv_path}...")
    df = pd.read_csv(train_csv_path)
    smiles_list = df['SMILES'].dropna().tolist()
    
    tokenizer = SMILESTokenizer()
    tokenizer.build_vocab(smiles_list)
    tokenizer.save_vocab(vocab_save_path)
    print(f"Saved SMILES vocab with {len(tokenizer.vocab)} tokens to {vocab_save_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build SMILES tokenizer vocabulary")
    parser.add_argument('--train_csv', type=str, default='dataset/BindingDB/train.csv')
    parser.add_argument('--vocab_out', type=str, default='dataset/BindingDB/smiles_vocab.json')
    args = parser.parse_args()
    
    if os.path.exists(args.train_csv):
        build_and_save_smiles_vocab(args.train_csv, args.vocab_out)
    else:
        print(f"Error: {args.train_csv} not found.")
