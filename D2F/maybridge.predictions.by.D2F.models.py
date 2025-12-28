#/ocean/projects/cis240078p/shared/.conda/envs/eddy01/bin/python

##### Maybridge prediction by BindingDB-Kd-DeNovo-full-trained & PDBbind2020-full-finetuned models (D2F) (with ray and jobarray) #####
#https://slurm.schedmd.com/job_array.html


##### Libraries #####
import os
PD = os.environ['SHARED']
TMP = os.environ['LOCAL']

print(PD, TMP)
print()

import sys
sys.path.append(PD+'/tools')
import DeepPurpose
from DeepPurpose import models

from joblib import Parallel, delayed
from tqdm_joblib import ParallelPbar

import pandas as pd
import numpy as np

import ray
print('> ray version:', ray.__version__)
ray.init(runtime_env={'working_dir': './', "py_modules": [DeepPurpose]})

import torch, joblib
print('> torch version:', torch.__version__)
print('> joblib version:', joblib.__version__)
print()


if __name__ == "__main__":
    from sys import argv
    w = int(argv[1])
    print(f"Sample Batch {w}")
    print()

    ##### START #####
    from time import time
    st = time()

    ##### Params and file names #####
    N_FOLDS = 5
    N_REPEATS = 10
    BATCH_SIZE = int(os.environ['BATCH_SIZE'])
    
    NAME_TARGET = 'ASGPR_5JQ1_1'
    SEQ_TARGET = PD+'/data/rcsb_pdb_5JQ1.fasta'

    MODEL_DIR1 = PD+'/models/regression/finetuned/BDB-full-PDBbind2020-refined-regsm-full/'
    MODEL_DIR2 = '/jobarray-RCV-'
    
    INPUT_FILE1 = PD+'/data/Maybridge_HitDiscover.smiles.unique.v2.csv'

    RESULT_FOLDER = './'
    OUTPUT_FILENAME0 = 'maybridge.predictions.by.D2F.models.'
    OUTPUT_FILENAME1 = RESULT_FOLDER + OUTPUT_FILENAME0 + 'all.batch'+str(w)+'.csv'

    ##### Load models #####
    MODELS = sorted([x for x in os.listdir(MODEL_DIR1) if not x.startswith('.')])
    
    def load_model(k):
        m = MODELS[k]
        tmp = []
        for i in range(N_REPEATS):
            for j in range(N_FOLDS):
                tmp_name = 'R' + str(i) + 'F' + str(j)
                tmp.append(models.model_pretrained(MODEL_DIR1 + m + MODEL_DIR2 + tmp_name))

        return {m:tmp}

    PRTMOD = ParallelPbar()(n_jobs=len(MODELS), prefer='threads')(
        delayed(load_model)(k) for k in range(len(MODELS))
    )

    PRTMOD = {list(x.keys())[0]: list(x.values())[0] for x in PRTMOD}

    ##### Load data #####
    metadata = pd.read_csv(INPUT_FILE1, index_col=0)

    lignames = metadata.Code
    smiles = metadata.SMILES.values

    target = pd.read_table(SEQ_TARGET).values[0]

    def batch(w):
        for i,j in enumerate(range(0, metadata.shape[0], BATCH_SIZE)):
            if i == w:
                tmp1 = lignames[j:(j+BATCH_SIZE)]
                tmp2 = smiles[j:(j+BATCH_SIZE)]
        return tmp1, tmp2

    lignames, smiles = batch(w)

    ##### Prediction #####
    PRTMOD = ray.put(PRTMOD)
    
    @ray.remote
    def PredictionUnit(r, PRTMOD):
        y_pred_all = []
        for i, (lig_name, smile) in enumerate(zip(lignames, smiles)):
            print(i)
            y_pred = {}
            for k,m in PRTMOD.items():
                tmp_pred = []
                tmp_names = []
                for f in range(N_FOLDS):
                    tmp_names.append('R' + str(r) + 'F' + str(f))
                    j = r*N_FOLDS + f
                    tmp_pred.append(models.repurpose([smile], target, m[j], [lig_name], NAME_TARGET, 
                                                     BindingDB=False, verbose=0, 
                                                     result_folder=TMP)[0])
    
                y_pred[k] = pd.DataFrame(tmp_pred, index=[k + '-' + n for n in tmp_names]).T
    
            y_pred_all.append(pd.concat(list(y_pred.values()), axis=1))
    
        os.system("find "+TMP+" -maxdepth 1 -name 'repurposing*.output.txt' -type f -delete")
    
        return(y_pred_all)

    #https://docs.ray.io/en/latest/ray-observability/user-guides/debug-apps/debug-memory.html
    ray_output = [PredictionUnit.options(memory=60*1024*1024*1024).remote(r, PRTMOD) for r in range(N_REPEATS)]
    y_pred_all = ray.get(ray_output)
    ray.shutdown()

    print()

    ##### Results #####
    pred_all = pd.concat([pd.concat(x, axis=0).reset_index(drop=True) for x in y_pred_all], axis=1)
    pred_all.insert(0, 'Ligand', lignames.values)

    pred_all.to_csv(OUTPUT_FILENAME1, index=False)
    
    ##### END #####
    time_taken = time() - st
    if time_taken < 60:
        print('Analysis time taken: %.2f sec.' %time_taken)
    elif time_taken < 3600:
        print('Analysis time taken: %.2f min.' %(time_taken / 60))
    else:
        print('Analysis time taken: %.2f hr.' %(time_taken / 3600))
