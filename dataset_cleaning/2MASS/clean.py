import pandas as pd
#def read_psc(path):
#    return df

def read_file(path, columns_path):
	columns = pd.read_csv(columns_path, header=None).iloc[:,0]
	def get_columns(line):
		return line.split()[0]
	columns = list(map(get_columns, columns))
	df = pd.read_csv(path, sep='|', names=columns)
	return df
xsc = read_file('/hpi/fs00/share/fg/naumann/lukas.laskowski/2MASS/samples/xsc/100_xsc.csv', 'xsc_columns')
psc = read_file('/hpi/fs00/share/fg/naumann/lukas.laskowski/2MASS/samples/100_2MASS.txt', 'psc_columns')
xsc['source'] = 'xsc'
psc['source'] = 'psc'
#rename columns for xsc
xsc.rename(columns={'j_msig_k20fe': 'j_msig',
                   'j_m_k20fe': 'j_m',
                   'k_m_k20fe': 'k_m',
                   'k_msig_k20fe': 'k_msig',
                   'h_m_k20fe': 'h_m',
                   'h_msig_k20fe': 'h_msig',
                   'jdate':'j_date'
                   },
          inplace=True, errors='raise')
psc = psc[['source', 'id', 'ra', 'dec','designation','glat', 'glon','hemis','date','j_date', 'j_m', 'j_msigcom', 'h_m', 'h_msigcom', 'k_m', 'k_msigcom']]
xsc = xsc[['source', 'id', 'ra', 'dec','designation','glat', 'glon','hemis','date','j_date', 'j_m', 'j_msigcom', 'h_m', 'h_msigcom', 'k_m', 'k_msigcom']]

all_stars = pd.concat([xsc, psc])
groundtruth = []
for row in all_stars[all_stars['source']=='psc'].iterrows(all_stars):
    xsc_row = all_stars[row['id'] == all_stars['id'] & all_stars['source'] == 'xsc']
    assert len(xsc_row) == 1
    idx_xsc = xsc_row.loc[:,0].index
    idx_psc = row.loc[:,0].index
    groundtruth.append((idx_xsc, idx_psc))
groundtruth = pd.DataFrame(groundtruth, columns=['xsc', 'psc'])
