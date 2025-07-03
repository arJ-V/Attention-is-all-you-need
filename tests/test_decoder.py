import torch
import unittest
from src.model.decoder import DecoderLayer, Decoder

class TestDecoder(unittest.TestCase):

    def test_decoder_layer(self):
        d_model = 512
        d_inner = 2048
        n_head = 8
        d_k = 64
        d_v = 64
        
        dec_layer = DecoderLayer(d_model, d_inner, n_head, d_k, d_v)
        
        # Test with a sample input
        dec_input = torch.randn(1, 10, d_model)
        enc_output = torch.randn(1, 10, d_model)
        
        output, slf_attn, dec_enc_attn = dec_layer(dec_input, enc_output)
        
        self.assertEqual(output.shape, dec_input.shape)
        self.assertEqual(slf_attn.shape, (1, n_head, 10, 10))
        self.assertEqual(dec_enc_attn.shape, (1, n_head, 10, 10))

    def test_decoder(self):
        n_trg_vocab = 1000
        d_word_vec = 512
        n_layers = 6
        n_head = 8
        d_k = 64
        d_v = 64
        d_model = 512
        d_inner = 2048
        pad_idx = 0
        
        decoder = Decoder(
            n_trg_vocab=n_trg_vocab, d_word_vec=d_word_vec, n_layers=n_layers,
            n_head=n_head, d_k=d_k, d_v=d_v, d_model=d_model, d_inner=d_inner,
            pad_idx=pad_idx
        )
        
        # Test with a sample input
        trg_seq = torch.randint(0, n_trg_vocab, (1, 10))
        trg_mask = (trg_seq != pad_idx).unsqueeze(-2)
        enc_output = torch.randn(1, 10, d_model)
        src_mask = torch.ones(1, 1, 10)
        
        output, *_ = decoder(trg_seq, trg_mask, enc_output, src_mask)
        
        self.assertEqual(output.shape, (1, 10, d_model))

if __name__ == '__main__':
    unittest.main()
