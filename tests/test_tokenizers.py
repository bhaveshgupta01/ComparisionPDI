import os
import tempfile
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from src.data.tokenizers import SMILESTokenizer, ProteinTokenizer

class TestSMILESTokenizer(unittest.TestCase):
    def setUp(self):
        self.tokenizer = SMILESTokenizer(max_len=10)
        self.smiles_list = ["CC(=O)Oc1ccccc1C(=O)O", "C1=CC=CC=C1"]

    def test_build_vocab(self):
        self.tokenizer.build_vocab(self.smiles_list)
        # Check special tokens
        self.assertIn('<pad>', self.tokenizer.vocab)
        self.assertIn('<cls>', self.tokenizer.vocab)
        self.assertIn('<sep>', self.tokenizer.vocab)
        self.assertIn('<unk>', self.tokenizer.vocab)
        
        # Check some expected tokens
        self.assertIn('C', self.tokenizer.vocab)
        self.assertIn('O', self.tokenizer.vocab)
        self.assertIn('c', self.tokenizer.vocab)
        self.assertIn('1', self.tokenizer.vocab)
        self.assertIn('=', self.tokenizer.vocab)

    def test_tokenize(self):
        tokens = self.tokenizer.tokenize(self.smiles_list[1]) # "C1=CC=CC=C1"
        expected_tokens = ['C', '1', '=', 'C', 'C', '=', 'C', 'C', '=', 'C', '1']
        self.assertEqual(tokens, expected_tokens)

    def test_encode_and_pad(self):
        self.tokenizer.build_vocab(self.smiles_list)
        # Using a short SMILES
        smiles = "CCO"
        encoded = self.tokenizer.encode(smiles)
        
        # max_len is 10
        self.assertEqual(len(encoded), 10)
        
        # Check start and sep tokens
        self.assertEqual(encoded[0], self.tokenizer.vocab['<cls>'])
        self.assertEqual(encoded[len(self.tokenizer.tokenize(smiles)) + 1], self.tokenizer.vocab['<sep>'])
        
        # Check padding
        pad_id = self.tokenizer.vocab['<pad>']
        for i in range(len(self.tokenizer.tokenize(smiles)) + 2, 10):
            self.assertEqual(encoded[i], pad_id)
            
    def test_encode_truncate(self):
        self.tokenizer.build_vocab(self.smiles_list)
        # Using a long SMILES to force truncation (max_len=10)
        smiles = "CC(=O)Oc1ccccc1C(=O)O"
        encoded = self.tokenizer.encode(smiles)
        
        self.assertEqual(len(encoded), 10)
        self.assertEqual(encoded[0], self.tokenizer.vocab['<cls>'])
        self.assertEqual(encoded[-1], self.tokenizer.vocab['<sep>'])

    def test_save_and_load_vocab(self):
        self.tokenizer.build_vocab(self.smiles_list)
        original_vocab = self.tokenizer.vocab.copy()
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp:
            tmp_path = tmp.name
            
        try:
            self.tokenizer.save_vocab(tmp_path)
            
            new_tokenizer = SMILESTokenizer()
            new_tokenizer.load_vocab(tmp_path)
            
            self.assertEqual(original_vocab, new_tokenizer.vocab)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


class TestProteinTokenizer(unittest.TestCase):
    def setUp(self):
        self.tokenizer = ProteinTokenizer(max_len=10)

    def test_tokenize(self):
        sequence = "ACDEFGHIKLMNPQRSTV"
        tokens = self.tokenizer.tokenize(sequence)
        self.assertEqual(tokens, list(sequence))

    def test_encode_pad(self):
        sequence = "ACD"
        encoded = self.tokenizer.encode(sequence)
        
        self.assertEqual(len(encoded), 10)
        self.assertEqual(encoded[0], self.tokenizer.vocab['<cls>'])
        self.assertEqual(encoded[4], self.tokenizer.vocab['<sep>'])
        
        # Check padding
        pad_id = self.tokenizer.vocab['<pad>']
        for i in range(5, 10):
            self.assertEqual(encoded[i], pad_id)
            
    def test_encode_truncate(self):
        sequence = "ACDEFGHIKLMNPQRSTV"
        encoded = self.tokenizer.encode(sequence)
        
        self.assertEqual(len(encoded), 10)
        self.assertEqual(encoded[0], self.tokenizer.vocab['<cls>'])
        self.assertEqual(encoded[-1], self.tokenizer.vocab['<sep>'])
        
    def test_unknown_token(self):
        sequence = "ACDZZZ"
        encoded = self.tokenizer.encode(sequence)
        
        # Expected tokens: <cls> A C D X X X <sep> <pad> <pad>
        self.assertEqual(encoded[0], self.tokenizer.vocab['<cls>'])
        self.assertEqual(encoded[1], self.tokenizer.vocab['A'])
        self.assertEqual(encoded[2], self.tokenizer.vocab['C'])
        self.assertEqual(encoded[3], self.tokenizer.vocab['D'])
        self.assertEqual(encoded[4], self.tokenizer.vocab['X'])
        self.assertEqual(encoded[5], self.tokenizer.vocab['X'])
        self.assertEqual(encoded[6], self.tokenizer.vocab['X'])
        self.assertEqual(encoded[7], self.tokenizer.vocab['<sep>'])

if __name__ == '__main__':
    unittest.main()
