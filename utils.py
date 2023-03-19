import os
import numpy as np
import torch

OFFSET = 0

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

def obtain_histograms(density_tep, binary_tep, degrees, bins):
        
        gh = np.array([])
        gsh = np.array([])
        dh = np.array([])
        dsh = np.array([])
        th = np.array([])
        tsh = np.array([])
        
        gstate_values = density_tep*(2*binary_tep-1)
        
        ## GLOBAL HISTOGRAM
        for bin in bins:
            global_histogram = np.histogram(density_tep, bins=bin, range=(OFFSET,1))[0]
            global_histogram = global_histogram/np.sum(global_histogram)
            
            if(len(gh)==0):
                gh = np.copy(global_histogram)
            else:
                gh = np.concatenate([gh, global_histogram])
                
        ## GLOBAL STATE HISTOGRAM
        for bin in bins:
            gstate_histogram = np.histogram(gstate_values, bins=bin, range=(-1+OFFSET,1))[0]
            gstate_histogram = gstate_histogram/np.sum(gstate_histogram)

            if(len(gsh)==0):
                gsh = np.copy(gstate_histogram)
            else:
                gsh = np.concatenate([gsh, gstate_histogram])
        
        ## DEGREE HISTOGRAM IMPLEMENTATION
        for bin in bins:

            max_degree = np.max(degrees)
            degree_histogram = []
            gstate_dhistogram = []
            for i in range(1,max_degree+1):
                indexes = np.where(degrees == i)
                if(len(indexes[0])):
                    dtep_values_degree = density_tep[:, indexes].reshape(-1)
                    degree_histogram.append(np.histogram(dtep_values_degree, bins=bin, range=(OFFSET,1))[0])
                    gstate_values_degree = gstate_values[:,indexes].reshape(-1)
                    gstate_dhistogram.append(np.histogram(gstate_values_degree, bins=bin, range=(-1+OFFSET,1))[0])
            degree_histogram = np.array(degree_histogram)
            degree_histogram = degree_histogram.mean(axis=0)
            degree_histogram = degree_histogram/np.sum(degree_histogram)
            gstate_dhistogram = np.array(gstate_dhistogram)
            gstate_dhistogram = gstate_dhistogram.mean(axis=0)
            gstate_dhistogram = gstate_dhistogram/np.sum(gstate_dhistogram)

            if(len(dh)==0):
                dh = np.copy(degree_histogram)
            else:
                dh = np.concatenate([dh, degree_histogram])

            if(len(dsh)==0):
                dsh = np.copy(gstate_dhistogram)
            else:
                dsh = np.concatenate([dsh, gstate_dhistogram])
                
        for bin in bins:
            ##TEMPORAL HISTOGRAM IMPLEMENTATION
            max_it = gstate_values.shape[0]
            temporal_dtep_hist = []
            temporal_gstate_hist = []
            for i in range(max_it):
                temporal_dtep_hist.append(np.histogram(density_tep[i,:], bins=bin, range=(OFFSET,1))[0])
                temporal_gstate_hist.append(np.histogram(gstate_values[i,:], bins=bin, range=(-1+OFFSET,1))[0])

            temporal_dtep_hist = np.array(temporal_dtep_hist)
            temporal_dtep_hist = temporal_dtep_hist.mean(axis=0)
            temporal_dtep_hist = temporal_dtep_hist/np.sum(temporal_dtep_hist)
            temporal_gstate_hist = np.array(temporal_gstate_hist)
            temporal_gstate_hist = temporal_gstate_hist.mean(axis=0)
            temporal_gstate_hist = temporal_gstate_hist/np.sum(temporal_gstate_hist)

            if(len(th)==0):
                th = np.copy(temporal_dtep_hist)
            else:
                th = np.concatenate([th, temporal_dtep_hist])

            if(len(tsh)==0):
                tsh = np.copy(temporal_gstate_hist)
            else:
                tsh = np.concatenate([tsh, temporal_gstate_hist])
        
        f = np.concatenate([gh, gsh, dh, dsh, th, tsh])
        
        t_f = torch.tensor(f)
                
        
        return t_f