import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tabix # look up fragments file from Cell Ranger
from scipy.io import mmread # loading sparse matrix from Cell Ranger
import anndata


def CreateEpiAnnData(fdir):
    '''Create an AnnData object with CellRanger output directory'''
    
    print('Loading matrix file..')
    X = mmread(fdir + 'matrix.mtx').toarray().T
    print('Done!')
    peaks = pd.read_csv(fdir + 'genes.tsv', header=None)[0].values
    barcodes = pd.read_csv(fdir + 'barcodes.tsv', header=None)[0].values
    
    AnnData = anndata.AnnData(X=X, var=peaks, obs=barcodes)
    print('Computing basic QC..')
    AnnData.obs['nfeatures'] = (AnnData.X>0).sum(axis=1)
    AnnData.obs['ncounts'] = AnnData.X.sum(axis=1)
    AnnData.var['ncells'] = (AnnData.X>0).sum(axis=0)
    AnnData.var['ncounts'] = AnnData.X.sum(axis=0)
    print('Done!')
    return AnnData

def plot_simple_qc(AnnData):
    '''Plots histograms of per-cell coverage, per-cell # of peaks, and number of cells in each peak'''
    fig, axs = plt.subplots(1,3, figsize=(20,4), dpi=100)
    axs[0].hist(AnnData.obs['ncounts'].values ,bins=30)
    axs[0].set_xlabel('counts')
    axs[0].set_ylabel('number of cells')

    axs[1].hist(AnnData.obs['nfeatures'].values, bins=30)
    axs[1].set_xlabel('open features')
    axs[1].set_ylabel('number of cells')

    axs[2].hist(AnnData.var['ncells'].values, bins=30)
    axs[2].set_xlabel('number of cells')
    axs[2].set_ylabel('number of features')

    plt.show()

def filter_cells_peaks(AnnData, min_counts, max_counts, min_cells):
    '''
    Filter cells and peaks (features) according to coverage
    
    Keyword arguments
    min_counts : minimum coverage per cell
    max_counts : maximum coverage per cell
    min_cells : minimum number of cells that a feature is non-zero
    '''
    AnnData = AnnData[AnnData.obs['ncounts'] >= min_counts]
    AnnData = AnnData[AnnData.obs['ncounts'] <= max_counts]
    AnnData = AnnData[:, AnnData.var['ncells']  > min_cells]
    return AnnData
    
# term-frequency inverse-document frequency normalization (not available in scanpy)
def tf_idf_normalization(AnnData, scale_factor=1e4):
    '''
    Computes term-frequency inverse-document frequency for scATAC-seq
    X : n rows (cells), p columns (peaks)
    return: Xnorm
    '''
    X = AnnData.X
    N = X.shape[0]
    npeaks = X.sum(axis=1, keepdims=True) #counts per cell n-vector
    TF = X / npeaks                       # normalize each cell, n by p
    IDF = N / X.sum(axis=0, keepdims=True)               # N / peak count across all cells, p-vector
    AnnData.X = np.log1p(TF*IDF*scale_factor)
    return AnnData

# extract region from tbx file
def read_tabix(tbx, region):
    '''Helper function for gene_activity_matrix()'''
    chrom, start, end = region
    record = tbx.query(str(chrom), start, end)
    fragment_df = pd.DataFrame([x for x in record])
    return fragment_df

# gene activities from scATAC-seq
def gene_activity_matrix(fragments, features, barcodes):
    '''
    Computes the activity of each feature in scATAC-seq
    fragments : fragment file that is bgzipped (provided by cellranger)
    features: chr, start, end, gene
    barcodes: list of barcodes
    returns: gene activity matrix
    '''
    tb  = tabix.open(fragments)    
    gene_activity = np.zeros((len(barcodes), len(features)))
    barcode_lookup = dict(zip(barcodes, np.arange(1, 1+len(barcodes)))) #hashmap to correspond barcodes with index in matrix

    for i in range(features.shape[0]):
        chrom, start, end = features.iloc[i, [0, 1, 2]]
        fragment_df = read_tabix(tb, (chrom, start, end))
        if fragment_df.shape[0] > 0:
            curr_barcodes = fragment_df[3].values            
            for b in curr_barcodes:
                z = barcode_lookup.get(b)
                if z:
                    gene_activity[z-1, i] += 1
        if i%1000 == 0 or i == features.shape[0]:
            percent_complete = str(np.round(100*((i+1)/features.shape[0]), decimals=2))
            print('\r Progress: '+percent_complete+'%', end="")

    gene_activity = pd.DataFrame(gene_activity, index=barcodes, columns=features['gene'].values)

    return gene_activity

def top_n_features(AnnData, q):
    '''
    Returns the features that have counts greater than the q-th quantile
    '''
    assert q >= 0 and q <= 1, 'Quantile must be between 0 and 1'

    q_quantile = np.quantile(AnnData.var['ncounts'].values, q)
    return AnnData[:, AnnData.var['ncounts'].values > q_quantile]
