import unittest
from unittest.mock import patch
from b_filter_comparison import filtered_value,write_filter


class FilterTest(unittest.TestCase):
    def test_only_filter_bits_change_and_restore(self):
        value=filtered_value(0x6161,15)
        self.assertEqual(value,0xF1F1)
        self.assertEqual(filtered_value(value,6),0x6161)
        for original,setting in [(0,15),(0x6161,0),(0x6160,15)]:
            with self.assertRaises(RuntimeError):filtered_value(original,setting)

    @patch('b_filter_comparison.rpc',return_value='running')
    @patch('b_filter_comparison.read',side_effect=[0,1,0,0])
    def test_active_pwm_prevents_any_write(self,read,rpc):
        with self.assertRaisesRegex(RuntimeError,'pwm_active'):
            write_filter(None,0x6161,0xF1F1,lambda *a:None)
        self.assertFalse(any('mww' in str(c) for c in rpc.call_args_list))

    @patch('b_filter_comparison.rpc',return_value='running')
    @patch('b_filter_comparison.read',side_effect=[0,0,0,0,0x6161,0xF1F1])
    def test_guarded_write_and_readback(self,read,rpc):
        write_filter(None,0x6161,0xF1F1,lambda *a:None)
        self.assertIn('mww 0x40000418 0xf1f1',[c.args[1] for c in rpc.call_args_list])


if __name__=='__main__':unittest.main()
