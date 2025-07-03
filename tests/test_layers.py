import torch
import unittest
from src.model.layers import PositionalEncoding, PositionwiseFeedForward

class TestLayers(unittest.TestCase):

    def test_positional_encoding(self):
        d_hid = 512
        n_position = 200
        pe = PositionalEncoding(d_hid, n_position)
        
        # Test with a sample input
        x = torch.randn(1, 10, d_hid)
        output = pe(x)
        
        self.assertEqual(output.shape, x.shape)
        # Check if positional encoding is added
        self.assertFalse(torch.all(torch.eq(output, x)))

    def test_positionwise_feed_forward(self):
        d_in = 512
        d_hid = 2048
        pff = PositionwiseFeedForward(d_in, d_hid)
        
        # Test with a sample input
        x = torch.randn(1, 10, d_in)
        output = pff(x)
        
        self.assertEqual(output.shape, x.shape)

if __name__ == '__main__':
    unittest.main()
