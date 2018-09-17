import argparse
import json
import logging
import numpy as np
import os
import time
import csv

from scoop import futures

import esnet

# Initialize logger
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

###############################################################################################
# The next part needs to be in the global scope, since all workers
# need access to these variables. I got pickling problems when using
# them as arguments in the evaluation function. I couldn't pickle the
# partial function for some reason, even though it should be supported.
############################################################################
# Parse input arguments
############################################################################
parser = argparse.ArgumentParser()
parser.add_argument("data", help="path to data file", type=str)
parser.add_argument("esnconfig", help="path to ESN config file", type=str)
parser.add_argument("reconstructconfig", help="path to reconstruct config file", type=str)
parser.add_argument("nexp", help="number of runs", type=int)
args = parser.parse_args()

############################################################################
# Read config file
############################################################################
config = json.load(open(args.esnconfig + '.json', 'r'))
reconstructconfig = json.load(open(args.reconstructconfig + '.json', 'r'))

############################################################################
# Load data
############################################################################
# If the data is stored in a directory, load the data from there. Otherwise,
# load from the single file and split it.

allPredictions = []
dataType = args.data.split('/')[-1]

if os.path.isdir(args.data):
    Xtr, Ytr, _, _, Xte, Yte, Yscaler = esnet.load_from_dir(args.data)

elif dataType=='SantaFe' or dataType=='Sunspots' or dataType=='Hongik' \
        or dataType=='GEFC' or dataType=='Mackey' or dataType=='SP500' \
        or dataType=='Rainfall' or dataType=='Temperature' \
        or dataType == 'MinTempMel' or dataType == 'SunSpotsZu'\
        or dataType == 'TempVancouver' or dataType=='TempSanFrancisco' \
        or dataType=='TempMontreal' or dataType=='TempNewYork' or dataType=='TempBoston':
    #Xtr, Ytr, _, _, Xte, Yte, Yscaler = esnet.generate_datasets_santafe(args.data)
    X, Y = esnet.load_from_text(args.data)

    # Construct training/test sets
    Xtr, Ytr, _, _, Xte, Yte, Yscaler = esnet.generate_datasets(X, Y)

    Xtr, Xte = esnet.reconstruct_input_1d([Xtr, Xte], reconstructconfig)
    Ytr, Yte = esnet.reconstruct_output_1d([Ytr, Yte], reconstructconfig)

elif dataType=='GEFC_temp':
    X, Y = esnet.load_from_text(args.data)

    # Construct training/test sets
    Xtr, Ytr, _, _, Xte, Yte, Yscaler = esnet.generate_datasets(X, Y)

    # Reconstruct
    Xtr, Xte = esnet.reconstruct_input_2d([Xtr, Xte], reconstructconfig)
    Ytr, Yte = esnet.reconstruct_output_2d([Ytr, Yte], reconstructconfig)

else:
    X, Y = esnet.load_from_text(args.data)

    # Construct training/test sets
    Xtr, Ytr, _, _, Xte, Yte, Yscaler = esnet.generate_datasets(X, Y)

    # Reconstruct
    Xtr, Xte = esnet.reconstruct_input_3d([Xtr, Xte], reconstructconfig)
    Ytr, Yte = esnet.reconstruct_output_3d([Ytr, Yte], reconstructconfig)

def single_run(dummy):
    """
    This function will be run by the workers.
    """
    start_time = time.time()
    predictions,error = esnet.run_from_config(Xtr, Ytr, Xte, Yte, config, Yscaler)
    run_time = time.time()-start_time

    predictions = Yscaler.inverse_transform(predictions)
    allPredictions.append(predictions)

    #return error
    return run_time

def exportPredictions(predictionResults):
    # Write predictions of all runs into files
    predictionConfig = args.esnconfig.split('/')[-1]
    predictionPath = './predictions/' + predictionConfig

    np.savetxt(predictionPath, np.column_stack(allPredictions), delimiter=',')

def main():
    for i in range(1100,1501,100):
        config['n_internal_units'] = i
        if ('n_dim' in config) and (config['n_dim'] != None):
            config['n_dim'] = int(0.1*i)
        if i>1000:
            args.esnconfig = args.esnconfig.replace(str(i-100),str(i))

        # Run in parallel and store result in a numpy array
        errors = np.array(list(map(single_run, range(args.nexp))), dtype=float)

        print("Errors:")
        print(errors)

        print("Mean:")
        print(np.mean(errors))

        print("Std:")
        print(np.std(errors))

        configName = args.esnconfig.split('/')[-1]
        dataName = args.data.split('/')[-1]
        runningInfo = [configName,dataName,np.mean(errors)]
        writePath = './results/running_time.csv'
        with open(writePath,'a') as f:
            writer = csv.writer(f)
            writer.writerow(runningInfo)

        real = Yscaler.inverse_transform(Yte)[100:]
        allPredictions.insert(0,real)

        exportPredictions(allPredictions)

if __name__ == "__main__":
    main()
