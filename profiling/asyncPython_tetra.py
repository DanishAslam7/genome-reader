from multiprocessing import Pool
import os
from pathlib import Path
import numpy as np
import pandas as pd

#SI=['r','x','v','s','aa','l','o','q','y']
#SD=['t','n','p','ma']

#EI=['ac','ad']
#ED=['ae']

CODE_DIR = Path(__file__).resolve().parent
TETRA_PARAM_FILE = Path(os.environ.get("TETRA_PARAM_FILE", CODE_DIR / "param_files" / "tetramer.csv"))
_DFRAME = None


def _profile_workers():
    workers = int(os.environ.get("PROFILE_WORKERS", os.cpu_count() or 1))
    return max(1, workers)


def _moving_window_size():
    return int(os.environ.get("PROFILE_MOVING_WINDOW", "25"))


def _param_frame():
    global _DFRAME
    if _DFRAME is None:
        _DFRAME = pd.read_csv(TETRA_PARAM_FILE, index_col=0)
    return _DFRAME


def normalize_params(sequence_list):
    print("starting normalisation of read sequences")
    sequence_list = [seq.strip().upper() for seq in sequence_list if seq]
    with Pool(processes=_profile_workers()) as pool:
        param_list = pool.map(calculateParameters, sequence_list)
    return param_list

def energyStructParamsMP(normalised_params_list):

    nml = len(normalised_params_list)
    print("Starting combining Energy and Struct")
    pool = Pool()
    SIParams_all_seq = pool.starmap(combineStructEnergyParams,[(SI,list(normalised_params_list[seq].items())) for seq in range(nml)])
    pool.close()
    pool.join()

    pool = Pool()
    SDParams_all_seq = pool.starmap(combineStructEnergyParams,[(SD,list(normalised_params_list[seq].items())) for seq in range(nml)])
    pool.close()
    pool.join()
    '''
    pool = Pool()
    EIparams_all_seq = pool.starmap(combineStructEnergyParams,[(EI,list(normalised_params_list[seq].items())) for seq in range(nml)])
    pool.close()
    pool.join()

    pool = Pool()
    EDParams_all_seq = pool.starmap(combineStructEnergyParams,[(ED,list(normalised_params_list[seq].items())) for seq in range(nml)])
    pool.close()
    pool.join()
    '''
    combined_params_map = dict(zip(['SIParams_all_seq','SDParams_all_seq'],[SIParams_all_seq,SDParams_all_seq]))
    print(type(combined_params_map))
    print(combined_params_map.keys())
    print(combined_params_map['SIParams_all_seq'][0][0:10])
    print(f"the length of SI_params is: {len(combined_params_map['SIParams_all_seq'])}")
    print(f"the length of first seq in SI_params is: {len(combined_params_map['SIParams_all_seq'][0])}")
    return transformStructEnerMap(combined_params_map)

def assign_params(param_map,plist):
    param_map['l'].append(plist['l'])
    param_map['ma'].append(plist['ma'])
    param_map['n'].append(plist['n'])
    param_map['o'].append(plist['o'])
    param_map['p'].append(plist['p'])
    param_map['q'].append(plist['q'])

    return param_map


def calculateMovingAverages(param_map):
    moving_win_size = _moving_window_size()
    moving_param_map = {}
    for k, v in param_map.items():
        arr = np.array(v)
        weights = np.ones(moving_win_size) / moving_win_size
        moving_averages = np.convolve(arr, weights, mode='valid')
        moving_param_map[k] = moving_averages.tolist()
    return normalizeMovingAverages(moving_param_map)


def normalizeMovingAverages(moving_param_map):
    normalized_map = {}
    for k, arr in moving_param_map.items():
        arr = np.array(arr)
        arr_min = arr.min()
        arr_max = arr.max()
        rang = arr_max - arr_min
        if rang == 0:
            normalized_arr = np.zeros_like(arr)
        else:
            normalized_arr = (arr - arr_min) / rang
        normalized_map[k] = normalized_arr.tolist()
    return normalized_map


def calculateParameters(sequence):
    sequence = sequence.strip().upper()
    param_map = {'l':[],'ma':[],'n':[],'o':[],'p':[],'q':[]}
    noofbases = len(sequence)
    dframe = _param_frame()
    if noofbases == 0:
        return
    trimotifs = []
    for m in range(noofbases - 3):
        trimotifs.append(sequence[m:m + 4])
    for motif in trimotifs:
        # m=str(motif, 'utf-8')
        if (motif in dframe.columns):
            assign_params(param_map, dframe[motif])

    return calculateMovingAverages(param_map)

def combineStructEnergyParams(array,normalized_list_tuples):
    normalized_map = dict(normalized_list_tuples)
    
    maps = np.zeros(len(normalized_map['r']))
    for k in array:
        arr = np.array(normalized_map[k])
        #for i in range(len(arr)):
        maps +=arr
    return maps

def transformStructEnerMap(struct_ener_map):
    transformed_map = {}
    for k in struct_ener_map.keys():
        values_of_all_seq_per_param = struct_ener_map[k]
        for i in range(len(values_of_all_seq_per_param)):
            try:
                transformed_map[i]
            except KeyError:
                transformed_map[i] = {}
            transformed_map[i][k] = values_of_all_seq_per_param[i]
    return transformed_map
