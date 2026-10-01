import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from compare_cv_methods import adapted_folds, FUNCTIONS, scores
from volatility_experiment import configuration


class CvComparisonTests(unittest.TestCase):
    def test_native_future_samples_removed_and_calendars_match(self):
        cfg=configuration()
        cfg['train_end_exclusive']='2020-05-01'
        cfg['validation_end_exclusive']='2020-07-01'
        grid=pd.date_range('2020-01-01',periods=270,tz='UTC')
        panel=pd.DataFrame(np.ones((270,4)),index=grid,columns=cfg['symbols'])
        methods={}; audits={}
        for name,function in FUNCTIONS.items():
            methods[name],audits[name]=adapted_folds(panel,cfg,function)
            for fold in methods[name]:
                self.assertLess(fold['train'].max()+7,fold['val'].min())
                self.assertLess(fold['val'].max()+7,fold['test'].min())
                self.assertEqual(len(fold['train']),len(set(fold['train'])))
        self.assertGreater(audits['KFold']['native_future_training_label_occurrences'],0)
        self.assertEqual(audits['ForwardChaining']['native_future_training_label_occurrences'],0)
        for f in range(5):
            for role in ['train','val','test']:
                np.testing.assert_array_equal(methods['KFold'][f][role],methods['ForwardChaining'][f][role])

    def test_score_formula(self):
        y=np.arange(70,dtype=float).reshape(10,7)+1
        p=y+3
        result=scores(y,p)
        expected=np.mean(1-((y-p)**2).sum(axis=0)/((y-y.mean(axis=0))**2).sum(axis=0))
        self.assertAlmostEqual(result['r2'],expected)
        self.assertEqual(result['rmse'],3)


if __name__=='__main__': unittest.main()
