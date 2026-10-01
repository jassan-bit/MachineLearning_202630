import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from minute_experiment import windows,RowSpace,estimator,predict_artifact
from volatility_experiment import fit_model
from run_minute_experiment import save_checkpoint


class MinuteTests(unittest.TestCase):
    def test_interrupted_save_preserves_previous_checkpoint(self):
        with TemporaryDirectory() as directory:
            path=Path(directory)/'checkpoint.joblib'
            save_checkpoint(path,{'completed':12})
            def interrupt(state,target,**kwargs):
                target.write_bytes(b'incomplete')
                raise OSError('Simulated interruption')
            with patch('run_minute_experiment.joblib.dump',side_effect=interrupt):
                with self.assertRaises(OSError): save_checkpoint(path,{'completed':24})
            self.assertEqual(joblib.load(path),{'completed':12})

    def test_every_minute_and_calendar_end(self):
        minutes=np.arange(40*1440,dtype=float)
        for lag in [7,14,21,28]:
            X=windows(minutes,lag,np.array([28,29]))
            self.assertEqual(X.shape,(2,lag*1440))
            self.assertEqual(X[0,0],(29-lag)*1440)
            self.assertEqual(X[0,-1],29*1440-1)
            np.testing.assert_array_equal(np.diff(X[0]),np.ones(lag*1440-1))
        minutes[28*1440+300]=np.nan
        with self.assertRaises(ValueError): windows(minutes,7,np.array([28]))

    def test_equivalent_full_minute_coefficients(self):
        rng=np.random.default_rng(6)
        X=rng.normal(size=(36,110)); test=rng.normal(size=(9,110)); y=rng.normal(size=(36,7))
        with threadpool_limits(limits=2):
            mapping=RowSpace().fit(X)
            compressed=fit_model(estimator(.1,.01),mapping.train,y)
            direct=fit_model(estimator(.1,.01),mapping.scaler.transform(X),y)
        a=compressed.predict(mapping.transform(test))
        np.testing.assert_allclose(a,direct.predict(mapping.scaler.transform(test)),atol=1e-4,rtol=1e-4)
        artifact=mapping.export(compressed)
        np.testing.assert_allclose(a,predict_artifact(artifact,test),atol=1e-10,rtol=1e-10)
        self.assertEqual(artifact['weights'].shape,(110,7))
        self.assertLess(mapping.gram_relative_error,1e-9)


if __name__=='__main__': unittest.main()
