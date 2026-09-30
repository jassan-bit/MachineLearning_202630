from pathlib import Path
import runpy,unittest
import numpy as np
module=runpy.run_path(str(Path(__file__).resolve().parents[1]/'src/22_change_events.py'))

class ChangePoints(unittest.TestCase):
    def test_exact_step_and_constant(self):
        k,gain=module['split_mean'](np.r_[np.zeros(120),np.ones(180)*3],90)
        self.assertEqual(k,120);self.assertAlmostEqual(gain,1)
        self.assertEqual(module['split_mean'](np.ones(300),90)[1],0)
    def test_offset_and_scale_invariance(self):
        x=np.random.default_rng(7).normal(size=400);x[210:]+=2
        k,gain=module['split_mean'](x,90)
        other,ratio=module['split_mean'](10+5*x,90)
        self.assertEqual(k,other);self.assertAlmostEqual(gain,ratio)
    def test_missing_values_are_rejected(self):
        with self.assertRaises(ValueError):module['split_mean']([1,np.nan,2],1)

if __name__=='__main__':unittest.main()
