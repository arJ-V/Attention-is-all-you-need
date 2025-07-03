import torch
import unittest
from src.model.transformer import Transformer

class TestTransformer(unittest.TestCase):

    def test_transformer(self):
        n_src_vocab = 1000
        n_trg_vocab = 1000
        src_pad_idx = 0
        trg_pad_idx = 0
        
        model = Transformer(
            n_src_vocab=n_src_vocab,
            n_trg_vocab=n_trg_vocab,
            src_pad_idx=src_pad_idx,
            trg_pad_idx=trg_pad_idx
        )
        
        # Test with a sample input
        src_seq = torch.randint(0, n_src_vocab, (1, 10))
        trg_seq = torch.randint(0, n_trg_vocab, (1, 12))
        
        output = model(src_seq, trg_seq)
        
        self.assertEqual(output.shape, (1 * 12, n_trg_vocab))

if __name__ == '__main__':
    unittest.main()
