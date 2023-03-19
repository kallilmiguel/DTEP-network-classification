import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="DTEP with RAE parameter specification")
    #data, paths, and other settings of general setup
    parser.add_argument('--dataset', type=str, default='kingdom', help="Database Graph")
    parser.add_argument('--tep_seed', type=str, default=15, help="Select seed to use as initial state of graph")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size to use (increase if you have VRAM available)")
    parser.add_argument('--gpu', type=int, default=0, help="Select specific GPU for execution")
    parser.add_argument('--base_seed', type=int, default=666999, help="Seed to use in the fold shuffling and classification procedures")
    parser.add_argument('--folds', type=int, default=10, help="Number of folds to use for classification")
    parser.add_argument('--reps', type=int, default=10, help="Number of repetitions for classification")
    
    return parser.parse_args()

#%%
import numpy as np
import datasets as ds
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split, KFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import random
import itertools

if __name__ == "__main__":
    
    args = parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = args.dataset
    repetitions = args.reps
    folds = args.folds
    base_seed = args.base_seed
    tep_seed = args.tep_seed
    
    bins = []
    stuff = [20,40,60,80,100]
    for L in range(1,3):
        for subset in itertools.combinations(stuff, L):
            bins.append(subset)
            
    for bin in bins:
        print(f"Bins: {bin}")

        graphs_ds = ds.Network(dataset, bins=bin, gpu=args.gpu, seed=tep_seed)
        graph_dataloader = DataLoader(dataset=graphs_ds, batch_size=64, shuffle=True, num_workers=0)

        feature_size = graphs_ds[0][0].shape[0]
        X = np.empty((0,feature_size))
        y = np.empty((0))

        for batch in graph_dataloader:
            
            data, labels = batch[0], batch[1]
            
            X = np.vstack((X, data.cpu().detach().numpy()))
            y = np.hstack((y, labels.cpu().numpy()))
            
        kfold = KFold(n_splits=folds, shuffle=True)
        accs_SVM, preds_SVM = [],[]

        for i in range(repetitions):
            seed = base_seed*(i+1)
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)
            
            for train_index, test_index in kfold.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                
                std = StandardScaler()
                X_train = std.fit_transform(X_train)
                X_test = std.transform(X_test)
                
                svm = SVC(C=1.0, kernel='linear', degree=3, gamma='scale', 
                                                coef0=0.0, shrinking=True, probability=False, tol=0.001,
                                                cache_size=200, class_weight=None, verbose=False, 
                                                max_iter=100000, decision_function_shape='ovr', 
                                                break_ties=False, random_state=seed)
                            
                            
                svm.fit(X_train,y_train)
                preds=svm.predict(X_test)            
                preds_SVM.append(preds)            
                acc= accuracy_score(y_test, preds)
                accs_SVM.append(acc*100)
                
                results = {
                    'accs_SVM': accs_SVM,
                    'preds_SVM': preds_SVM,
                }
                
        if repetitions > 1:
            svm = []
            for it_ in range(repetitions):
                svm.append(np.mean(results['accs_SVM'][it_*folds: it_*folds + folds]))   
            results['accs_SVM'] = svm
                        
            print(f'Acc_SVM:', f"{np.round(np.mean(results['accs_SVM']), 1):.1f} (+-{np.round(np.std(results['accs_SVM']), 1):.1f})", sep=' ', flush=True)

