import torch
import unittest
from src.model.encoder import EncoderLayer, Encoder

class TestEncoder(unittest.TestCase):

    def test_encoder_layer(self):
        d_model = 512
        d_inner = 2048
        n_head = 8
        d_k = 64
        d_v = 64
        
        enc_layer = EncoderLayer(d_model, d_inner, n_head, d_k, d_v)
        
        # Test with a sample input
        x = torch.randn(1, 10, d_model)
        output, attn = enc_layer(x)
        
        self.assertEqual(output.shape, x.shape)
        self.assertEqual(attn.shape, (1, n_head, 10, 10))

    def test_encoder(self):
        n_src_vocab = 1000
        d_word_vec = 512
        n_layers = 6
        n_head = 8
        d_k = 64
        d_v = 64
        d_model = 512
        d_inner = 2048
        pad_idx = 0
        
        encoder = Encoder(
            n_src_vocab=n_src_vocab, d_word_vec=d_word_vec, n_layers=n_layers,
            n_head=n_head, d_k=d_k, d_v=d_v, d_model=d_model, d_inner=d_inner,
            pad_idx=pad_idx
        )
        
        # Test with a sample input
        src_seq = torch.randint(0, n_src_vocab, (1, 10))
        src_mask = (src_seq != pad_idx).unsqueeze(-2)
        
        output, *_ = encoder(src_seq, src_mask)
        
        self.assertEqual(output.shape, (1, 10, d_model))

if __name__ == '__main__':
    unittest.main()
