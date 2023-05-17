from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import pdist, euclidean
import pandas as pd
import numpy as np

def perform_similarity_calculation(df, filename, columns):
    print(f"Start similarity calculation for {filename}")
    df = normalization(df, columns)
    calculate_similarity_matrix(df, filename)

def normalization(df, columns):
    scaler = MinMaxScaler()
    print("Start normalization")
    df[columns] = scaler.fit_transform(df[columns])
    print("Normalization done")
    return df[columns]

def similarity_func(u, v):
    u_mask = ~np.isnan(u)
    u = u[u_mask]
    v = v[u_mask]
    v_mask = ~np.isnan(v)
    u = u[v_mask]
    v = v[v_mask]
    return 1/(1+euclidean(u,v))

def calculate_similarity_matrix(df, filename):
    print("Start similarity calculation")
    df = df.to_numpy().astype(np.float)
    
    dists = pdist(df, similarity_func)
    print("Similarity calculation done")
    np.save(f'/hpi/fs00/share/fg-naumann/lukas.laskowski/datasets/{filename}/similarity.npy', dists)
 
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
    for dataset in config:
        print(dataset['name'])
        df = pd.read_csv(f'{base_path}{dataset["name"]}/features.csv')
        for c in dataset['columns']:
            if df[c].dtype.kind not in 'iufcb':
                print("PROBLEM: ", c, df[c].dtype.kind)
                break
        perform_similarity_calculation(df, dataset['name'], dataset['columns'])