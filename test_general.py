"""Software fixtures ONLY; none of these numbers are experimental observations."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ml_general as ml

class GeneralTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='ml_software_test_')
        self.root = Path(self.tmp.name)
        rng = np.random.default_rng(31415)
        labels = np.repeat(['Test_A', 'Test_B', 'Test_C'], [8, 10, 12])
        X = rng.normal(size=(len(labels), 4)) + np.repeat([0, .8, 1.3], [8, 10, 12])[:, None]
        self.frame = pd.DataFrame(X, columns=['f1', 'f2', 'f3', 'f4'])
        self.frame.insert(0, 'label', labels)
        self.frame.insert(0, 'id', [f'TEST_ONLY_{i:03d}' for i in range(len(X))])
        self.path = self.root / 'software_fixture.csv'
        self.frame.to_csv(self.path, index=False)
        self.dataset = {'key': 'test', 'source': {'path': str(self.path)}, 'sample_id_column': 'id', 'label_column': 'label', 'feature_columns': ['f1','f2','f3','f4']}
        self.config = {'random_seed': 20261009, 'validation': {'outer_folds': 2, 'inner_folds': 2},
                       'svm': {'C_grid': [.1,1], 'gamma_grid': ['scale']}, 'plots': {'enabled': False, 'grid_size': 30}}
        self.config_path = self.root / 'config.json'
        self.config_path.write_text('{}')
        self.out = self.root / 'out'

    def tearDown(self):
        self.tmp.cleanup()

    def run_case(self, dataset=None, config=None):
        return ml.analyze(dataset or self.dataset, config or self.config, self.config_path, self.out, False, {}, 'powershell.exe')

    def test_arbitrary_class_sizes_and_fold_local_transforms(self):
        config = {**self.config, 'preprocessing': {'select_k': 3, 'pca_components': 2}}
        before = ml.sha(self.path)
        s = self.run_case(config=config)
        self.assertEqual(s['n_samples'], 30)
        predictions = pd.read_csv(self.out/'test/predictions.csv')
        self.assertEqual(len(predictions), 30)
        for a in s['algorithms']:
            self.assertEqual(sum(map(sum,s['metrics'][a]['confusion'])), 30)
            self.assertEqual(s['metrics'][a]['correct'], int(predictions[f'{a}_Correct'].sum()))
        records = ml.load_json(self.out/'test/fold_parameters.json')
        partitions = pd.read_csv(self.out/'test/cv_partitions.csv')
        for record in records:
            train_ids = partitions[(partitions.Outer_Fold == record['outer_fold']) & (partitions.Role == 'train')].Sample_ID
            train = self.frame.set_index('id').loc[train_ids, ['f1','f2','f3','f4']].to_numpy()
            np.testing.assert_allclose(record['scale_mean'], train.mean(axis=0), rtol=0, atol=1e-14)
            self.assertEqual(len(record['selected_features']), 3)
        self.assertEqual(before, ml.sha(self.path))
        excel = pd.read_excel(self.out/'test/results.xlsx',sheet_name='InputData')
        np.testing.assert_allclose(excel[['f1','f2','f3','f4']], self.frame[['f1','f2','f3','f4']], rtol=0, atol=1e-14)

    def test_existing_scores_are_not_recomputed(self):
        d = {**self.dataset, 'input_kind': 'existing_pca', 'feature_columns': ['f1','f2'], 'pca_explained_percent': [60,30]}
        self.run_case(dataset=d)
        pred = pd.read_csv(self.out/'test/predictions.csv', float_precision='round_trip')
        np.testing.assert_array_equal(pred[['PC1','PC2']], self.frame[['f1','f2']].to_numpy())

    def test_group_separation_outer_and_inner(self):
        labels = np.repeat(['Test_A', 'Test_B', 'Test_C'], 12)
        rng = np.random.default_rng(19)
        frame = pd.DataFrame(rng.normal(size=(36,4)),columns=['f1','f2','f3','f4'])
        frame.insert(0,'label',labels)
        frame.insert(0,'id',[f'TEST_ONLY_G{i}' for i in range(36)])
        frame['prep'] = [f'PREP_{a}_{i//2}' for a in range(3) for i in range(12)]
        frame.to_csv(self.path,index=False)
        self.run_case(dataset={**self.dataset,'preparation_group_column':'prep'})
        outer = pd.read_csv(self.out/'test/cv_partitions.csv')
        inner = pd.read_csv(self.out/'test/inner_partitions.csv')
        for _, part in outer.groupby('Outer_Fold'):
            self.assertFalse(set(part[part.Role=='train'].Preparation_Group) & set(part[part.Role=='validation'].Preparation_Group))
        for _, part in inner.groupby(['Outer_Fold','Inner_Fold']):
            self.assertFalse(set(part[part.Role=='train'].Preparation_Group) & set(part[part.Role=='validation'].Preparation_Group))

    def test_binary_lda_has_no_fake_ld2(self):
        self.frame = self.frame[self.frame.label != 'Test_C']
        self.frame.to_csv(self.path,index=False)
        s=self.run_case()
        p=next(x for x in s['plots'] if x['mode']=='LDA')
        self.assertTrue(p['one_dimensional'])
        self.assertEqual(p['y_column'],'Group_ID')
        predictions=pd.read_csv(self.out/'test/predictions.csv')
        self.assertTrue(predictions.LD2.isna().all())

    def test_missing_default_stops_without_deleting_rows(self):
        self.frame.loc[0,'f1']=np.nan
        self.frame.to_csv(self.path,index=False)
        with self.assertRaisesRegex(ValueError,'Missing numeric inputs'):
            self.run_case()

    def test_median_is_fold_local_and_source_missing_retained(self):
        self.frame.loc[0,'f1']=np.nan
        self.frame.to_csv(self.path,index=False)
        self.run_case(config={**self.config,'preprocessing':{'missing':'median'}})
        source=pd.read_csv(self.out/'test/input_data.csv')
        self.assertTrue(pd.isna(source.loc[0,'f1']))
        records=ml.load_json(self.out/'test/fold_parameters.json')
        partitions=pd.read_csv(self.out/'test/cv_partitions.csv')
        for record in records:
            train_ids=partitions[(partitions.Outer_Fold==record['outer_fold']) & (partitions.Role=='train')].Sample_ID
            train=self.frame.set_index('id').loc[train_ids,['f1','f2','f3','f4']].to_numpy()
            np.testing.assert_allclose(record['median_values'],np.nanmedian(train,axis=0),rtol=0,atol=1e-14)

    def test_transpose_requires_real_metadata_mapping(self):
        spectral=pd.DataFrame({'Wavelength_nm':[300,400,500,600]})
        for _, row in self.frame.iterrows():
            spectral[row['id']]=[row[f] for f in ['f1','f2','f3','f4']]
        spectral.to_csv(self.path,index=False)
        meta=self.root/'metadata.csv'
        self.frame[['id','label']].to_csv(meta,index=False)
        d={**self.dataset,'orientation':'samples_in_columns','feature_axis_column':'Wavelength_nm','sample_columns':self.frame.id.tolist(),
           'metadata':{'path':str(meta),'sample_column':'id'},'feature_columns':['300','400','500','600']}
        self.run_case(dataset=d)
        x=pd.read_csv(self.out/'test/input_data.csv',float_precision='round_trip')
        np.testing.assert_array_equal(x[['300','400','500','600']],self.frame[['f1','f2','f3','f4']].to_numpy())

    def test_duplicate_ids_and_leaky_features_are_rejected(self):
        frame=self.frame.copy()
        frame.loc[1,'id']=frame.loc[0,'id']
        source={'path':str(self.path),'type':'table','sheet':''}
        with self.assertRaisesRegex(ValueError,'Repeated sample IDs'):
            ml.input_matrix(self.dataset,frame,source,self.root)
        with self.assertRaisesRegex(ValueError,'must not be classifier features'):
            ml.input_matrix({**self.dataset,'feature_columns':['label']},self.frame,source,self.root)

    def test_insufficient_folds_are_not_changed(self):
        with self.assertRaisesRegex(ValueError,'insufficient'):
            ml.cv_splits(np.arange(10).reshape(5,2),np.array([1,1,2,2,2]),None,3,20261009)

    def test_check_only_does_not_fit_models(self):
        result=ml.analyze(self.dataset,self.config,self.config_path,self.out,True,{},'powershell.exe')
        self.assertTrue(result['checked_only'])
        self.assertFalse(list((self.out/'test').glob('*.joblib')))

if __name__ == '__main__':
    unittest.main()
