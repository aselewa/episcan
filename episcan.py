import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import anndata # data structure for single cell omics
import tabix
from scipy.io import mmread 
import utils

import umap
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import LatentDirichletAllocation

def CreateEpiAnnData(fdir):
    '''Create an AnnData object with CellRanger output directory'''
    
    assert os.path.isfile(fdir+'matrix.mtx'), 'Matrix file not found'
    assert os.path.isfile(fdir+'peaks.tsv'), 'Peaks file not found'
    assert os.path.isfile(fdir+'barcodes.tsv'), 'Barcodes file not found'

    print('Loading matrix file..')
    X = mmread(fdir + 'matrix.mtx').toarray().T
    print('Done!')
    peaks = pd.read_csv(fdir + 'peaks.tsv', header=None)[0].values
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


def top_n_features(AnnData, q):
    '''
    Returns the features that have counts greater than the q-th quantile
    '''
    assert q >= 0 and q <= 1, 'Quantile must be between 0 and 1'

    q_quantile = np.quantile(AnnData.var['ncounts'].values, q)
    return AnnData[:, AnnData.var['ncounts'].values > q_quantile]

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

def run_svd(AnnData, n_components = 15):
    '''
    Perform truncated SVD on tf-idf normalized data
    n_components: number of components to compute SVD for
    Returns: AnnData
    '''
    X = AnnData.X
    svd = TruncatedSVD(algorithm='arpack', n_components = n_components, n_iter=7, random_state=42)    
    X_reduced = svd.fit_transform(X)
    X_reduced = (X_reduced - X_reduced.mean(axis=0))/X_reduced.std(axis=0)
    AnnData.obsm['X_pca'] = X_reduced
    AnnData.uns['explained_variance_ratio'] = svd.explained_variance_ratio_
    
    return AnnData

def run_umap(AnnData, n_pcs=15, n_neighbors=15, min_dist=0.5, n_components=2, metric='cosine'):
    '''
    UMAP Wrapper
    Computes UMAP from SVD reduced data
    Parameters are the same as those in UMAP() 
    Returns: AnnData
    '''
    X = AnnData.obsm['X_pca'][:,0:n_pcs]
    fit = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, metric=metric)
    u = fit.fit_transform(X)
    AnnData.obsm['X_umap'] = u
    
    return AnnData

def plot_umap(AnnData, size=1.5, ax=None, colors='spectral'):
    
    u = AnnData.obsm['X_umap']
    c = AnnData.obs[colors]
    if ax:
        ax.scatter(u[:,0], u[:,1], c = c, s=size)
    else:
        fig, ax = plt.subplots(1, 1, figsize=(8,6), dpi=100, s=size)
        scatter = ax.scatter(u[:,0], u[:,1], c = c)
        
    uniq = np.unique(c)
    if len(uniq) < 10:
        ax[i,j].legend(*scatter.legend_elements(), bbox_to_anchor=(1.1, 0.5), loc='center')
    else:
        fig.colorbar(scatter, ax=ax[i,j])
    ax.set_title(colors)
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')

def plot_multi_umap(AnnData, size=1.5, colors=['spectral']):
    
    u = AnnData.obsm['X_umap']
    nplots = len(colors)
    ncols = 2
    nrows = int(np.ceil(nplots/ncols)) + 1
         
    fig, ax = plt.subplots(nrows, ncols, figsize=(10,4*nrows), dpi=100)
    
    for k in range(nplots):
        i = k//2
        j = k%2
        c = AnnData.obs[colors[k]]
        scatter = ax[i,j].scatter(u[:,0], u[:,1], c = c, s = size)
        ax[i,j].set_title(colors[k])
        ax[i,j].set_xlabel('UMAP 1')
        ax[i,j].set_ylabel('UMAP 2')
        
        uniq = np.unique(c)
        if len(uniq) < 10:
            ax[i,j].legend(*scatter.legend_elements(), bbox_to_anchor=(1.1, 0.5), loc='center')
        else:
            fig.colorbar(scatter, ax=ax[i,j])
    
    fig.delaxes(ax[nrows-1, 0])
    fig.delaxes(ax[nrows-1, 1])
    
    if nplots % 2 != 0:
        fig.delaxes(ax[nrows-2, 1])
        
    fig.tight_layout()
                
            
def run_spectral_clustering(AnnData, n_pcs=15, n_clusters=3, n_neighbors=15):
    '''
    SpectralClustering wrapper
    Performs Spectral Clustering on Cells using SVD reduced data
    n_clusters : number of clusters
    n_neighbors : number of nearest neighbors in affinity graph
    returns: AnnData 
    '''
    X = AnnData.obsm['X_pca'][:,0:n_pcs]
    clustering = SpectralClustering(n_clusters=n_clusters, 
                                    eigen_solver=None, 
                                    affinity = 'nearest_neighbors',
                                    n_neighbors = n_neighbors,
                                    assign_labels="discretize", 
                                    random_state=0)
    clustering.fit(X)
    AnnData.obs['spectral'] = clustering.labels_
    
    return AnnData


def rn_lda(AnnData):
    '''Incomplete'''
    print('This function is incomplete.')
    X = AnnData.X
    lda = LatentDirichletAllocation(n_components=5, random_state=0)
    X_LDA = lda.fit_transform(X)
    AnnData.obsm['X_lda'] = X_lda
    return AnnData


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
        fragment_df = utils.read_tabix(tb, (chrom, start, end))
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

