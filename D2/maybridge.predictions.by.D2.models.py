#!/ocean/projects/cis240078p/shared/.conda/envs/eddy01/bin/python

##### Maybridge prediction by BindingDB-2020m2-fulldata de novo-trained models (D2) (with jobarray) #####
#https://slurm.schedmd.com/job_array.html


##### Libraries #####
import os
PD = os.environ['SHARED']
TMP = os.environ['LOCAL']

print(PD, TMP)
print()

import sys
sys.path.append(PD+'/tools')
from DeepPurpose import models

from joblib import Parallel, delayed

import pandas as pd
import numpy as np


##### Prediction Function #####
def PredictionUnit(r):
    y_pred = []
    for idx, (name_ligand, ligand) in enumerate(zip(lignames, smiles)):
        print(str(idx))
        
        m = PRTMOD[MODELS[r]]
        y_pred.append(models.repurpose([ligand], target, m, [name_ligand], NAME_TARGET,
                                       BindingDB=False, verbose=0,
                                       result_folder=TMP)[0])

    os.system("find "+TMP+" -maxdepth 1 -name 'repurposing*.output.txt' -type f -delete")
    
    return({MODELS[r]: y_pred})


if __name__ == "__main__":
    from sys import argv
    w = int(argv[1])
    print(f"Sample Batch {w}")
    print()

    ##### START #####
    from time import time
    st = time()

    import torch, joblib
    print('> torch version:', torch.__version__)
    print('> joblib version:', joblib.__version__)
    print()

    ##### Params and file names #####
    N_FOLDS = 5
    N_REPEATS = 10
    BATCH_SIZE = int(os.environ['BATCH_SIZE'])
    
    NAME_TARGET = 'ASGPR_5JQ1_1'
    SEQ_TARGET = PD+'/data/rcsb_pdb_5JQ1.fasta'

    MODEL_DIR = PD+'/models/regression/BindingDB-fulldata/'
    INPUT_FILE1 = PD+'/data/Maybridge_HitDiscover.smiles.unique.v2.csv'

    RESULT_FOLDER = './'
    OUTPUT_FILENAME0 = 'maybridge.predictions.by.D2.models.'
    OUTPUT_FILENAME1 = RESULT_FOLDER + OUTPUT_FILENAME0 + 'all.batch'+str(w)+'.csv'

    ##### Load models #####
    MODELS = sorted(os.listdir(MODEL_DIR))

    PRTMOD = {}
    for m in MODELS:
        PRTMOD[m] = models.model_pretrained(MODEL_DIR + m)

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
    y_pred_all = Parallel(n_jobs=len(MODELS),
                          prefer='threads',
                          verbose=1)(delayed(PredictionUnit)(r) for r in range(len(MODELS)))

    print()

    ##### Results #####
    pred_all = pd.concat([pd.DataFrame(x) for x in y_pred_all], axis=1)
    pred_all.insert(0, 'Ligand', lignames.values)

    pred_all.to_csv(OUTPUT_FILENAME1, index=False)
    
    print(pred_all.drop(['Ligand'], axis=1).corr().round(3))
    print()
    
    ##### END #####
    time_taken = time() - st
    if time_taken < 60:
        print('Analysis time taken: %.2f sec.' %time_taken)
    elif time_taken < 3600:
        print('Analysis time taken: %.2f min.' %(time_taken / 60))
    else:
        print('Analysis time taken: %.2f hr.' %(time_taken / 3600))
