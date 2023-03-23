import os
import torch
import pathlib
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import utils
import pickle

def getListOfFiles(dirName):
    # create a list of all files in a root dir
    listOfFile = os.listdir(dirName)
    allFiles = list()
    for entry in listOfFile:
        fullPath = os.path.join(dirName, entry)
        if ".DS" not in fullPath:
            if os.path.isdir(fullPath):
                allFiles = allFiles + getListOfFiles(fullPath)
            else:
                allFiles.append(fullPath)
                
    return allFiles

DATASETS_PATH = "/home/DATA/datasets/networks/"
TEP_OUTPUT_PATH = "/home/DATA/results/DTEP/TEPs/"
FEATURE_OUTPUT_PATH = "/home/DATA/results/DTEP/features/"

DATASETS = {
    '4models': DATASETS_PATH + 'synthetic/synthetic_model',
    'noise10': DATASETS_PATH + 'synthetic/synthetic_noise/ruido=10',
    'noise20': DATASETS_PATH + 'synthetic/synthetic_noise/ruido=20',
    'noise30': DATASETS_PATH + 'synthetic/synthetic_noise/ruido=30',
    'scalefree': DATASETS_PATH + 'synthetic/synthetic_scalefree',
    'kingdom': DATASETS_PATH + 'real/kingdom',
    'animals': DATASETS_PATH + 'real/animals',
    'fungi': DATASETS_PATH + 'real/fungi',
    'plant': DATASETS_PATH + 'real/plant',
    'firmicutes-bacillis': DATASETS_PATH + 'real/firmicutes-bacillis',
    'actinobacteria': DATASETS_PATH + 'real/actinobacteria',
}

CLASSES = {
    '4models': ['barabasi', 'erdos', 'geo', 'watts'],
    'noise10': ['BANL15_', 'BANL2_', 'BANL5_', 'BA_', 'ER_', 'MEN_', 'GEO_', 'WS_'],
    'noise20': ['BANL15_', 'BANL2_', 'BANL5_', 'BA_', 'ER_', 'MEN_', 'GEO_', 'WS_'],
    'noise30': ['BANL15_', 'BANL2_', 'BANL5_', 'BA_', 'ER_', 'MEN_', 'GEO_', 'WS_'],
    'scalefree': ['barabasi', 'mendes', 'nonlinear05', 'nonlinear15', 'nonlinear2'],
    'kingdom': ['animals', 'fungi', 'plants', 'protist'],
    'animals': ['fishes', 'birds', 'insects', 'mammals'],
    'fungi': ['basidiomycetes', 'eurotiomycetes', 'saccharomycetes', 'sordariomycetes'],
    'plant': ['eudicots', 'greenalgae', 'monocots'],
    'firmicutes-bacillis': ['Bacillus', 'Lactobacillus', 'Staphylococcus', 'Streptococcus'],
    'actinobacteria': ['Corynebacterium', 'Mycobacterium', 'Streptomyces'],
}

RULES_PATH = "./rules/"
RULES = {
    '4models': RULES_PATH + '4models.txt',
    'noise10': RULES_PATH + 'ns10.txt',
    'noise20': RULES_PATH + 'ns20.txt',
    'noise30': RULES_PATH + 'ns30.txt',
    'scalefree': RULES_PATH + 'scalefree.txt',
    'kingdom': RULES_PATH + 'kingdom.txt',
    'animals': RULES_PATH + 'animal.txt',
    'fungi': RULES_PATH + 'fungi.txt',
    'plant': RULES_PATH + 'plant.txt',
    'firmicutes-bacillis': RULES_PATH + 'firmicutes.txt',
    'actinobacteria': RULES_PATH + 'actinobacteria.txt',
}

SEED_PATH = "./seeds/"

class Network():
    
    def __init__(
        self,
        dataset: str,
        bins: list,
        seed: int = 4,
        gpu: int = 0
    ) -> None:
        
        self.bins = bins
        self._base_folder = pathlib.Path(DATASETS[dataset])
        self._meta_folder = self._base_folder / "labels"
        self._graph_folder = self._base_folder / "graphs"
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.feature_size = 6*sum(self.bins)
        
        if not self._check_exists:
            raise RuntimeError("Dataset not found.")
        
        self.gpu = gpu
        self.dataset = dataset
        self._graph_files = []
        classes = []
        self.names = []
        with open(self._meta_folder / 'classes.txt') as file:
            for line in file:
                cl, path = line.split(" ")
                filename = path.split("/")[-1]
                
                self._graph_files.append(path[:-1])
                self.names.append(filename[:-1])
                classes.append(cl)
                
        self.name_to_idx = dict(zip(self.names, range(len(self.names))))
        self.classes = sorted(set(classes))
        self.class_to_idx = dict(zip(self.classes, range(len(self.classes))))
        self._labels = [self.class_to_idx[cl] for cl in classes]
        self._graph_file_to_idx = dict(zip(self._graph_files, range(len(self._graph_files))))

        self._rule_file = RULES[self.dataset]
        self._seed_file = SEED_PATH + 'init_state_' + str(seed) + '.csv'
        self._tep_output_path = TEP_OUTPUT_PATH + dataset + '/'  
        self._feature_output_path = FEATURE_OUTPUT_PATH + dataset + '/' 
        
        for cl in classes:
            if not os.path.exists(self._tep_output_path + cl):
                os.makedirs(self._tep_output_path + cl)
            if not os.path.exists(self._feature_output_path + cl):
                os.makedirs(self._feature_output_path + cl)
    
    def __len__(self) -> int:
        return len(self._graph_files)
    
    def __getitem__(self, idx):
        
        file_name, label = self.names[idx], self._labels[idx]
        cl = self.classes[label]
            
        if not (self._check_tep_exist(file_name, cl)):
                self._save_tep(file_name, cl)
        
        if not (self._check_feature_exist(file_name, cl)):
            self._save_feature(file_name, cl)
            
        with open(self._feature_output_path + cl + '/' + file_name.split(".txt")[0] + f'_features_b={self.bins}.pkl', "rb") as f:
            data = pickle.load(f).to(self.device) 
        
        label = torch.tensor(label).to(self.device)
        
        return data, label
    
    def _check_exists(self) -> bool:
        return os.path.exists(self._base_folder) and os.path.isdir(self._base_folder)
        
    def _save_tep(self, file_name, label):
        
        os.system(f"CUDA_VISIBLE_DEVICES={self.gpu} /home/kallilzie/DTEP-network-classification/run_llna {file_name} {DATASETS[self.dataset]}/graphs/ {self._rule_file} {self._tep_output_path}{label}/ {self._seed_file}")
        
        return
    
    def _check_tep_exist(self, file_name, label) -> bool:
        
        if os.path.isfile(f'{self._tep_output_path}{label}/{file_name.split(".txt")[0]}_net_degree.csv'):
            return True
            
        return False
    
    def _check_feature_exist(self, file_name, label) -> bool:
        
        if os.path.isfile(f'{self._feature_output_path}{label}/{file_name.split(".txt")[0]}_features_b={str(self.bins)}.pkl'):
            return True
        
        return False
    
    def _save_feature(self, file_name, label):
        
        try:
            df_density = pd.read_csv(self._tep_output_path + label + '/' + file_name.split('.txt')[0] + '_net_rule_1_density.csv')
            df_binary = pd.read_csv(self._tep_output_path + label + '/' + file_name.split('.txt')[0] + '_net_rule_1_binary.csv')
            df_degree = pd.read_csv(self._tep_output_path + label + '/' + file_name.split('.txt')[0] + '_net_degree.csv', header=None)
            
            density_arr = df_density.iloc[:,:].values
            binary_arr = df_binary.iloc[:,:].values
            degrees_arr = df_degree.iloc[:,:].values
            
            X = utils.obtain_histograms(density_arr, binary_arr, degrees_arr, self.bins)
        
        except:
            
            X = torch.zeros(self.feature_size)
        
        
        with open(f'{self._feature_output_path}{label}/{file_name.split(".txt")[0]}_features_b={self.bins}.pkl', "wb") as f:
            pickle.dump(X, f)
            
        return
        
    
    def _set_bins(self, bins):
        self.bins = bins
        return
    