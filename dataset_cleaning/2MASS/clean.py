import pandas as pd
import os
#def read_psc(path):
#    return df

def read_file(path, columns_path):
	columns = pd.read_csv(columns_path, header=None).iloc[:,0]
	def get_columns(line):
		return line.split()[0]
	columns = list(map(get_columns, columns))
	print(columns)
	df = pd.read_csv(path, sep='|', names=columns)
	return df
base_path = "/hpi/fs00/share/fg-naumann/lukas.laskowski/2MASS/samples/"
xsc = read_file(os.path.join(base_path, 'xsc', '100_xsc'), 'xsc_columns')
psc = read_file(os.path.join(base_path, '100_2MASS.txt'), 'psc_columns')
#print(psc.columns)
#print(xsc.columns)
xsc['source'] = 'xsc'
psc['source'] = 'psc'
#rename columns for xsc
#xsc.drop(['ext_key'], inplace=True, axis=1)
xsc.rename(columns={'j_msig_k20fe': 'j_msigcom',
                   'j_m_k20fe': 'j_m',
                   'k_m_k20fe': 'k_m',
                   'k_msig_k20fe': 'k_msigcom',
                   'h_m_k20fe': 'h_m',
                   'h_msig_k20fe': 'h_msigcom',
                   #'scan_key': 'ext_key',
                   },
          inplace=True, errors='raise')
#xsc.to_csv('temp.csv', index=False)
print("DSds")
print(psc[psc['ext_key']==2000451]['ra'])
print("HDS")
#print("D)WDWDW")
psc = psc[['source','ext_key', 'ra', 'decl','designation','glat', 'glon','hemis','date','jdate', 'j_m', 'j_msigcom', 'h_m', 'h_msigcom', 'k_m', 'k_msigcom']]
#return
xsc = xsc[['source', 'ext_key','ra', 'decl','designation','glat', 'glon','hemis','date','jdate', 'j_m', 'j_msigcom', 'h_m', 'h_msigcom', 'k_m', 'k_msigcom']]

all_stars = pd.concat([xsc, psc])
groundtruth = []
for idx,row in psc.iterrows():
    if row['ext_key'] == 54361:
        #continue
        pass
    xsc_row = all_stars[row['ext_key'] == all_stars['ext_key']]
    xsc_row = xsc_row[xsc_row['source'] == 'xsc']
    #print(all_stars[all_stars['source'] == 'xsc'])
    assert len(xsc_row) == 1
    groundtruth.append((xsc_row.index[0], idx))
groundtruth = pd.DataFrame(groundtruth, columns=['xsc', 'psc'])
groundtruth.to_csv('groundtruth.csv', index=False)
all_stars.to_csv('all_stars.csv', index=False)
