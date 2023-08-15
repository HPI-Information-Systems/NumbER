from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import pdist, euclidean, cosine
import pandas as pd
import numpy as np
import os
import traceback
import time
from numba import jit
from NumbER.matching_solutions.matching_solutions.embitto import EmbittoMatchingSolution
#from NumbER.matching_solutions.utils.calculate_cosine import cosine
import re
import scipy
from sklearn.feature_extraction.text import TfidfVectorizer


def ngrams(string, n=3):
    string = re.sub(r'[,-./]|\sBD',r'', string)
    ngrams = zip(*[string[i:] for i in range(n)])
    return [''.join(ngram) for ngram in ngrams]

def perform_similarity_calculation(numerical_df, textual_df, filename):
    print(f"Start similarity calculation for {filename}")
    numerical_df.dropna(axis=1, how='all', inplace=True)
    numerical_df = normalization(numerical_df) if len(numerical_df.columns) > 0 else None
    calculate_similarity_matrix(numerical_df, textual_df, filename)

def normalization(df):
    scaler = MinMaxScaler()
    print("Start normalization")
    df = scaler.fit_transform(df)
    print("Normalization done")
    return df
@jit(nopython=True)
def cosine_similarity_numba(u:np.ndarray, v:np.ndarray):
    assert(u.shape[0] == v.shape[0])
    uv = 0
    uu = 0
    vv = 0
    for i in range(u.shape[0]):
        uv += u[i]*v[i]
        uu += u[i]*u[i]
        vv += v[i]*v[i]
    cos_theta = 1
    if uu!=0 and vv!=0:
        cos_theta = uv/np.sqrt(uu*vv)
    return cos_theta
@jit#(nopython=True)
def numerical_similarity_func(u, v):
    u_mask = ~np.isnan(u)
    u = u[u_mask]
    v = v[u_mask]
    v_mask = ~np.isnan(v)
    u = u[v_mask]
    v = v[v_mask]
    return cosine_similarity_numba(u, v)
    return 1 - np.dot(u, v)/(np.linalg.norm(u)*np.linalg.norm(v))
    return 1/(1+euclidean(u,v))

def textual_similarity_func(u, v):
    u_mask = ~np.isnan(u)
    u = u[u_mask]
    v = v[u_mask]
    v_mask = ~np.isnan(v)
    u = u[v_mask]
    v = v[v_mask]
    return 


def calculate_similarity_matrix(numerical_df, textual_df, filename):
    print("Start similarity calculation")
    if textual_df is not None:
        textual_df.dropna(axis=1, how='all', inplace=True)
        similarities = []
        #numerical_df = numerical_df.to_numpy().astype(np.float)
        for col in textual_df.columns:
            vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams)
            print(textual_df[col].values.astype('U'))
            print("Col", col)
            if col == "n_max":
                continue
            values = vectorizer.fit_transform(textual_df[col].values.astype('U')).toarray()
            start_time = time.time()
            sim = pdist(values, numerical_similarity_func)
            print("--- %s seconds ---" % (time.time() - start_time))
            similarities.append(sim)
            #textual_sim = len(similarities) * np.mean(similarities, axis=0) if len(similarities) > 0
        print("Textual similarity calculation done")
    else:
        print("No textual columns")
    if numerical_df is not None:
        numerical_sim = pdist(numerical_df, numerical_similarity_func)
        print("Numerical similarity calculation done")
        numerical_length = np.shape(numerical_df)[1]
        numerical_sim = numerical_length * numerical_sim
    if numerical_df is None:
        total = np.mean(similarities, axis=0)
    else:
        total = numerical_sim if textual_df is None else (numerical_length * numerical_sim + len(similarities) * np.mean(similarities, axis=0)) / (numerical_length + len(similarities))
    # total = (numerical_length * numerical_sim + len(similarities) * np.mean(similarities, axis=0)) / (numerical_length + len(similarities))
    print("Similarity calculation done", np.shape(total))
    #textual_df = textual_df.to_numpy()
    #cosine_a = cosine(scipy.sparse.csr_matrix(numerical_df), scipy.sparse.csr_matrix(numerical_df), 10)
    #dists = pdist(df, similarity_func)
    # print("Similarity calculation done")
    np.save(f'/hpi/fs00/share/fg-naumann/lukas.laskowski/datasets/{filename}/similarity_cosine.npy', total)
 
if __name__ == '__main__':
    base_path = '/hpi/fs00/share/fg-naumann/lukas.laskowski/datasets/'
    config = [
        {'name': 'protein', 'columns': ["volume","enclosure","surface","lipoSurface","depth","surf/vol","lid/hull","ellVol","ell_c/a","ell_b/a","surfGPs","lidGPs","hullGPs","siteAtms","accept","donor","aromat","hydrophobicity","metal","Cs","Ns","Os","Ss","Xs","acidicAA","basicAA","polarAA","apolarAA","sumAA","ALA","ARG","ASN","ASP","CYS"]},
		# {'name': 'vsx', 'columns': ["RAdeg", "DEdeg", "max", "min", "Epoch", "Period"]},
		# {'name': 'earthquakes', 'columns': ["depth","depth_uncertainty","horizontal_uncertainty","used_phase_count","used_station_count","standard_error","azimuthal_gap","minimum_distance","mag_value","mag_uncertainty","mag_station_count"]},
		# #{'name': 'protein', 'columns': []},
		# #{'name': '2MASS', 'columns': []},
		# {'name': 'baby_products_numeric', 'columns': ["width","height","length","weight_lb","weight_oz","price"]},
		# {'name': 'books3_numeric', 'columns': ["Price","Pages","height","width","length"]},
		# {'name': 'books3_numeric_no_isbn', 'columns': ["Price","Pages","height","width","length"]},
		# #{'name': 'books4_numeric', 'columns': ["Price", "height","width","length"]},	
		# {'name': 'x2_numeric', 'columns': ["weight_lb","weight_oz","weight_kg","weight_g","brand_refine","cpu_brand_refine","core_refine","frequency_refine","storage_refine","ram_refine"]},
		# {'name': 'x3_numeric', 'columns': ["weight_lb","weight_oz","weight_kg","weight_g","brand_refine","cpu_brand_refine","core_refine","frequency_refine","storage_refine","ram_refine"]},
	]
    # for dataset in config:
    #     print(dataset['name'])
    #     df = pd.read_csv(f'{base_path}{dataset["name"]}/features.csv')
    #     textual_columns = EmbittoMatchingSolution.get_textual_columns(df, False)
    #     numerical_columns = EmbittoMatchingSolution.get_numeric_columns(df)
    #     numerical_df = df[numerical_columns]
    #     textual_df = df[textual_columns]
    #     numerical_df.drop(columns=['id'], inplace=True)
    #     perform_similarity_calculation(df, dataset['name'], dataset['columns'])
    for dataset in ["baby_products_all"]:#os.listdir('/hpi/fs00/share/fg-naumann/lukas.laskowski/datasets/'):
        try:
            if os.path.exists(f'/hpi/fs00/share/fg-naumann/lukas.laskowski/datasets/{dataset}/similarity_cosine.npy') or dataset in ["books4_all", "2MASS", "books4_numeric", "baby_products_all_VORSICHT_ID"]:#
                print("Already done", dataset)
                continue
            print("Start with", dataset)
            df = pd.read_csv(f'/hpi/fs00/share/fg-naumann/lukas.laskowski/datasets/{dataset}/features.csv')
            textual_columns = EmbittoMatchingSolution.get_textual_columns(df, False)
            numerical_columns = EmbittoMatchingSolution.get_numeric_columns(df)
            numerical_df = df[numerical_columns] if len(numerical_columns) > 0 else None
            if textual_columns is not None:
                textual_df = df[textual_columns] if len(textual_columns) > 0 else None
                textual_df.drop(columns=['id'], inplace=True) if "id" in textual_df.columns else None
            else:
                textual_df = None
            numerical_df.drop(columns=['id'], inplace=True) if "id" in numerical_df.columns else None
            perform_similarity_calculation(numerical_df, textual_df, dataset)
        except Exception as e:
            print(e)
            traceback.print_exc()
            print("Error with", dataset)
            continue