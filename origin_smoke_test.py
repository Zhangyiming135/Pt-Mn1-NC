"""Optional Windows/Origin integration test using explicitly artificial fixtures."""
from pathlib import Path
import argparse
import json
import subprocess
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import ml_general as ml

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,help='A NEW empty software-test directory.')
    parser.add_argument('--powershell',default='powershell.exe')
    args=parser.parse_args()
    target=Path(args.output).resolve()
    if target.exists():
        raise FileExistsError('Use a new test directory; no deletion/overwrite is performed.')
    target.mkdir(parents=True)
    rng=np.random.default_rng(771)
    labels=np.repeat(['Software_Test_A','Software_Test_B','Software_Test_C'],[8,10,12])
    features=['f1','f2','f3','f4']
    frame=pd.DataFrame(rng.normal(size=(30,4))+np.repeat([0,.8,1.4],[8,10,12])[:,None],columns=features)
    frame.insert(0,'label',labels);frame.insert(0,'id',[f'TEST_ONLY_{i:03d}' for i in range(30)])
    frame.to_csv(target/'ARTIFICIAL_SOFTWARE_TEST_ONLY.csv',index=False)
    config={'output_root':str(target/'results'), 'random_seed':20261009,
            'validation':{'outer_folds':2,'inner_folds':2},'svm':{'C_grid':[.1,1],'gamma_grid':['scale']},
            'plots':{'enabled':True,'grid_size':40},
            'datasets':[{'key':'software_test','title':'SOFTWARE TEST ONLY - NOT EXPERIMENTAL',
                         'source':{'path':str(target/'ARTIFICIAL_SOFTWARE_TEST_ONLY.csv')},
                         'sample_id_column':'id','label_column':'label','feature_columns':features}]}
    cfg=target/'test_config.json';ml.save_json(cfg,config)
    subprocess.run([sys.executable,str(ROOT/'ml_general.py'),'--config',str(cfg),'--origin','--powershell',args.powershell],check=True)
    summary=ml.load_json(target/'results/software_test/summary.json')
    export=ml.load_json(target/'results/software_test/origin_export.json')
    # Test OPJU source reading from the saved native input sheet; do not train again.
    reread={'output_root':str(target/'origin_readback'),'datasets':[{'key':'readback',
            'source':{'type':'origin','path':export['project'],'worksheet':f"[{export['books']['InputData']}]Sheet1"},
            'sample_id_column':'A','label_column':'D','feature_columns':['G','H','I','J']}]}
    cfg2=target/'readback_config.json';ml.save_json(cfg2,reread)
    subprocess.run([sys.executable,str(ROOT/'ml_general.py'),'--config',str(cfg2),'--check-only','--powershell',args.powershell],check=True)
    original=pd.read_csv(target/'results/software_test/input_data.csv',float_precision='round_trip')
    readback=pd.read_csv(target/'origin_readback/readback/input_data.csv',float_precision='round_trip')
    np.testing.assert_array_equal(original[features],readback[['G','H','I','J']])
    assert original.True_Class.tolist()==readback.True_Class.tolist()
    # Exercise appending to an existing source project with colliding page names.
    append_config={'output_root':str(target/'origin_append'), 'algorithms':['LDA'],
                   'validation':{'outer_folds':2,'inner_folds':2}, 'plots':{'enabled':False},
                   'datasets':[dict(reread['datasets'][0],key='source_copy_test')]}
    cfg3=target/'append_config.json';ml.save_json(cfg3,append_config)
    subprocess.run([sys.executable,str(ROOT/'ml_general.py'),'--config',str(cfg3),'--origin','--powershell',args.powershell],check=True)
    verification=ml.load_json(target/'origin_append/source_copy_test/origin_verification.json')
    assert verification['pass'] and verification['source_origin_cells_checked']>0
    report={'software_test_only':True,'experimental_results':False,'native_origin_reopen':True,
            'input_features_roundtrip_identical':True,'source_labels_roundtrip_identical':True,
            'source_project_append_and_name_collision_verified':True,
            'integration_result':str(target/'results/software_test/origin_verification.json'),
            'scope':'Artificial fixtures only. No historical user dataset was run.'}
    ml.save_json(target/'integration_test_record.json',report)
    print('Origin integration and source-reading roundtrip passed (SOFTWARE TEST ONLY).')

if __name__=='__main__':
    main()
