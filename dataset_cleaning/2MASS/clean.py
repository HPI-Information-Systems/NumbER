import pandas as pd
#def read_psc(path):
#    return df

def read_file(path, columns_path):
	columns = pd.read_csv('xsc_columns.csv', header=None).iloc[:,0]
	def get_columns(line):
		return line.split()[0]
	columns = list(map(get_columns, columns))
	df = pd.read_csv(path, sep='|', names=columns)
	return df
xsc = read_file('xsc_sample.csv', 'xsc_columns.csv')
psc = read_file('psc_sample.csv', 'psc_columns.csv')

