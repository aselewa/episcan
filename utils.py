import os
import numpy as np
import pandas as pd
import tabix
from scipy.io import mmwrite

CHROM_HG38_LENS = {'chr1':248956422,'chr10':133797422,
'chr11':135086622,'chr12':133275309,
'chr13':114364328,'chr14':107043718,
'chr15':101991189,'chr16':90338345,
'chr17':83257441,'chr18':80373285,
'chr19':58617616,'chr2':242193529,
'chr20':64444167,'chr21':46709983,
'chr22':50818468,'chr3':198295559,
'chr4':190214555,'chr5':181538259,
'chr6':170805979,'chr7':159345973,
'chr8':145138636,'chr9':138394717}

def read_tabix(tbx, region):
    '''Helper function for gene_activity_matrix()'''
    chrom, start, end = region
    record = tbx.query(str(chrom), start, end)
    record = pd.DataFrame([x for x in record])
    return record

def make_bins(chrom_lens=CHROM_HG38_LENS, bin_size=500):
    
    bins = []
    for chrom,L in chrom_lens.items():
        if bin_size > L:
            n_bins = 1
        else:
            n_bins = L//bin_size
        bin_start = np.arange(1, n_bins*bin_size, bin_size)
        bins += zip([chrom]*n_bins, bin_start, bin_start+bin_size)
    return bins

def make_atac_matrix(fragments, barcodes, bin_size=5000):
    
    tb  = tabix.open(fragments)
    bins = make_bins(bin_size=bin_size)
    nbins = len(bins)
    ncells = len(barcodes)
    cell_map = dict(zip(barcodes, np.arange(1,ncells+1)))
    mtx = []
    
    for i,b in enumerate(bins):
        record = read_tabix(tb, b)
        if record.shape[0] > 0:
            curr_barcodes = record[3].values
            #uniq_bc, c = np.unique(curr_barcodes, return_counts=True)
            for j,bc in enumerate(curr_barcodes):
                z = cell_map.get(bc)
                if z:
                    entry = (i, z, c[j])
                    mtx.append(entry)
        percent_complete = str(np.round(100*((i+1)/len(bins)), decimals=2))
        print('\r Progress: '+percent_complete+'%', end="")
        
    return mtx