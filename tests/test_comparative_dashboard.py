"""Integration checks for actual aligned artifacts and dashboard callbacks."""
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from volatility_dashboard import data_loader as loader
from volatility_dashboard.metrics import aggregate
from volatility_dashboard.app import app
from volatility_dashboard.tabs import eda, modelos


class ComparativeDashboardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictions, cls.metrics, cls.audit = loader.repository()

    def test_all_seven_models_have_identical_test_keys_and_actuals(self):
        self.assertEqual(set(loader.FAMILIES), {'k-NN', 'Ridge', 'Lasso', 'Random Forest', 'XGBoost', 'SVR Lineal', 'HAR-Ridge + XGBoost'})
        self.assertEqual(set(self.predictions.model),set(loader.FAMILIES))
        reference = None
        for model,frame in self.predictions.groupby('model'):
            series = frame.set_index(loader.KEYS).actual.sort_index()
            self.assertEqual(len(series),40096)
            self.assertEqual(frame.origin.nunique(),358)
            if reference is not None:
                pd.testing.assert_series_equal(series,reference)
            reference = series

    def test_macro_metrics_match_saved_verified_results(self):
        for model,family in loader.FAMILIES.items():
            observed = self.metrics[self.metrics.model==model][['mae','rmse','mse','r2']].mean()
            saved = pd.read_csv(loader.ROOT/'results'/family/'macro_metrics.csv')
            name = 'HAR_Ridge_XGBoost' if model == 'HAR-Ridge + XGBoost' else None
            saved = saved[(saved.symbol=='GLOBAL_MACRO') & (saved.model.eq(name) if name else saved.model!='Persistence')].iloc[0]
            np.testing.assert_allclose(observed.to_numpy(float),saved[observed.index].to_numpy(float),rtol=1e-10)

    def test_corrupted_forecast_is_excluded_without_poisoning_reference(self):
        original = pd.read_csv
        def read(path,*args,**kwargs):
            frame = original(path,*args,**kwargs)
            if 'optimized_minute_knn_2023_2025' in str(path) and str(path).endswith('predictions.csv.gz'):
                frame.loc[0,'knn'] += .5
            return frame
        loader.repository.cache_clear()
        try:
            with patch.object(loader.pd,'read_csv',side_effect=read):
                predictions,_,audit = loader.repository()
            self.assertNotIn('k-NN',predictions.model.unique())
            self.assertEqual(len(predictions.model.unique()),6)
            self.assertEqual(audit.set_index('model').loc['k-NN','estado_comparable'],'Excluido')
        finally:
            loader.repository.cache_clear()

    def test_window_and_horizon_filters_do_not_mix_targets(self):
        view = aggregate(self.metrics,'BTCUSDT',14,3,['XGBoost','Ridge'])
        expected = self.metrics.query('symbol == "BTCUSDT" and volatility_window == 14 and horizon == 3 and model in ["XGBoost","Ridge"]')
        np.testing.assert_allclose(view.set_index('model').sort_index().rmse,expected.set_index('model').sort_index().rmse)

    def test_http_routes_and_three_tabs(self):
        client = app.server.test_client()
        for route in ['/','/healthz','/_dash-layout','/_dash-dependencies','/assets/style.css','/book-report']:
            with client.get(route) as response:
                self.assertEqual(response.status_code,200,route)
        self.assertEqual(client.get('/healthz').get_json(), {'status': 'ok'})
        layout = client.get('/_dash-layout').get_json()
        tabs = layout['props']['children'][1]['props']['children']
        self.assertEqual(len(tabs),3)
        for tab in ['contexto','eda','modelos']:
            response = client.post('/_dash-update-component',json={
                'output':'tab-content.children','outputs':{'id':'tab-content','property':'children'},
                'inputs':[{'id':'main-tabs','property':'value','value':tab}],
                'state':[],'changedPropIds':['main-tabs.value']})
            self.assertEqual(response.status_code,200,tab)
            self.assertNotIn('Datos no disponibles',response.get_data(as_text=True))

    def test_real_eda_and_models_callbacks_serialize(self):
        from plotly.utils import PlotlyJSONEncoder
        import json
        views = [
            eda.render('BTCUSDT',7,1,loader.FEATURES[0],'Spearman','2023-01-01','2024-12-24',['BTCUSDT','ETHUSDT']),
            eda.render('XRPUSDT',28,7,loader.FEATURES[3],'Pearson','2025-01-01','2025-12-31',list(loader.dataset().columns)),
            modelos.render('TODOS',7,'TODOS',list(loader.FAMILIES),'rmse','XGBoost','BTCUSDT',1),
            modelos.render('ETHUSDT',14,3,['Ridge','Lasso','SVR Lineal'],'r2','SVR Lineal','BTCUSDT',1),
            modelos.render('TODOS',7,'TODOS',list(loader.FAMILIES),'r2','HAR-Ridge + XGBoost','BTCUSDT',1),
        ]
        for view in views:
            self.assertGreater(len(json.dumps(view,cls=PlotlyJSONEncoder)),1000)


if __name__=='__main__':
    unittest.main()
