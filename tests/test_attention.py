import torch
import unittest
from src.model.attention import ScaledDotProductAttention, MultiHeadAttention

class TestAttention(unittest.TestCase):

    def test_scaled_dot_product_attention(self):
        temp_key = torch.randn(2, 4, 64)
        temp_value = torch.randn(2, 4, 64)
        temp_query = torch.randn(2, 4, 64)
        
        attn = ScaledDotProductAttention(temperature=64**0.5)
        output, attn_weights = attn(temp_query, temp_key, temp_value)
        
        self.assertEqual(output.shape, (2, 4, 64))
        self.assertEqual(attn_weights.shape, (2, 4, 4))

    def test_multi_head_attention(self):
        temp_key = torch.randn(2, 4, 512)
        temp_value = torch.randn(2, 4, 512)
        temp_query = torch.randn(2, 4, 512)
        
        mha = MultiHeadAttention(n_head=8, d_model=512, d_k=64, d_v=64)
        output, attn_weights = mha(temp_query, temp_key, temp_value)
        
        self.assertEqual(output.shape, (2, 4, 512))
        self.assertEqual(attn_weights.shape, (2, 8, 4, 4))

if __name__ == '__main__':
    unittest.main()
