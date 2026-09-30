"""Comprobaciones numéricas con estructuras de dimensionalidad conocida."""
from pathlib import Path
import importlib.util
import unittest
import numpy as np
import pandas as pd

spec=importlib.util.spec_from_file_location('multivariate',Path(__file__).resolve().parents[1]/'src/12_multivariate_close.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class MultivariateDiagnostics(unittest.TestCase):
    def test_duplicate_columns_have_one_effective_dimension(self):
        x=np.linspace(-3,3,100)
        X=pd.DataFrame({f'lag_{k}':x*(k+1)+k for k in range(4)},index=pd.date_range('2020-01-01',periods=100,freq='h'))
        stats,scaler,pca,corr,scores,flags=module.diagnose(X,trees=10,max_samples=32)
        self.assertAlmostEqual(stats['pc1_ratio'],1)
        self.assertAlmostEqual(stats['effective_rank'],1)
        self.assertEqual(stats['k95'],1)
        self.assertEqual(stats['numerical_rank'],1)
        self.assertTrue(np.isinf(stats['condition_number']))
        np.testing.assert_allclose(corr,1)
        np.testing.assert_array_equal(flags,scores>np.quantile(scores,.99))

    def test_equal_orthogonal_directions_retain_two_dimensions(self):
        angle=np.arange(128)*2*np.pi/128
        X=pd.DataFrame({'lag_0':np.sin(angle),'lag_1':np.cos(angle)},index=pd.date_range('2020-01-01',periods=128,freq='h'))
        stats,*_=module.diagnose(X,trees=10,max_samples=32)
        self.assertAlmostEqual(stats['pc1_ratio'],.5)
        self.assertAlmostEqual(stats['effective_rank'],2)
        self.assertEqual(stats['k95'],2)

    def test_target_or_constant_column_is_rejected(self):
        with self.assertRaises(ValueError):module.diagnose(pd.DataFrame({'lag_0':[1,2,3],'y':[1,2,3]}))
        with self.assertRaises(ValueError):module.diagnose(pd.DataFrame({'lag_0':[1,2,3],'lag_1':[1,1,1]}))


if __name__=='__main__':unittest.main()
