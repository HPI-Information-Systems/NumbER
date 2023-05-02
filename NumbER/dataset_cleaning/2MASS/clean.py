import pandas as pd
import sys
import os
#def read_psc(path):
#    return df

def read_file(path, columns_path):
	columns = pd.read_csv(columns_path, header=None).iloc[:,0]
	def get_columns(line):
		return line.split()[0]
	columns = list(map(get_columns, columns))
	print(columns)
	df = pd.read_csv(path, sep=',', names=columns)
	return df
base_path = "/hpi/fs00/share/fg-naumann/lukas.laskowski/2MASS/sampled_dataset/"
xsc = read_file(os.path.join(base_path, 'xsc.csv'), 'xsc_columns')
psc = read_file(os.path.join(base_path, 'psc_filtered.csv'), 'psc_columns')

xsc['source'] = 'xsc'
psc['source'] = 'psc'
#rename columns for xsc
#xsc.drop(['ext_key'], inplace=True, axis=1)
psc[psc["ext_key"]=="\\N"] = -1
psc["ext_key"] = psc["ext_key"].astype(int)

#xsc["ext_key"]=xsc["ext_key"].astype(str)
xsc.rename(columns={'j_msig_k20fe': 'j_msigcom',
                   'j_m_k20fe': 'j_m',
                   'k_m_k20fe': 'k_m',
                   'k_msig_k20fe': 'k_msigcom',
                   'h_m_k20fe': 'h_m',
                   'h_msig_k20fe': 'h_msigcom',
                   #'scan_key': 'ext_key',
                   },
          inplace=True, errors='raise')
psc = psc[['source','ext_key', 'ra', 'decl','designation','glat', 'glon','hemis','date','jdate', 'j_m', 'j_msigcom', 'h_m', 'h_msigcom', 'k_m', 'k_msigcom']]
#return
xsc = xsc[['source', 'ext_key','ra', 'decl','designation','glat', 'glon','hemis','date','jdate', 'j_m', 'j_msigcom', 'h_m', 'h_msigcom', 'k_m', 'k_msigcom']]

all_stars = pd.concat([xsc, psc])
all_stars.reset_index(inplace=True, drop=True)
groundtruth = []
for idx,row in psc.iterrows():
    if row['ext_key'] == -1:
        continue
    if idx % 10000 == 0:
        #continue
        print(idx, file=sys.stdout)
    rows = all_stars[row['ext_key'] == all_stars['ext_key']]
    psc_row = rows[rows['source'] == 'psc']
    xsc_row = rows[rows['source'] == 'xsc']
    #print(all_stars[all_stars['source'] == 'xsc'])
    assert len(xsc_row) == 1
    groundtruth.append((xsc_row.index[0], psc_row.index[0]))
groundtruth = pd.DataFrame(groundtruth, columns=['p1', 'p2'])#p1 is xsc, p2 is psc
groundtruth['prediction'] = 1
groundtruth.to_csv(os.path.join(base_path, 'groundtruth.csv'), index=False)
all_stars['id'] = all_stars.index
all_stars.drop(columns=['ext_key', 'source'], inplace=True)
all_stars.to_csv(os.path.join(base_path,'all_stars.csv'), index=False)
