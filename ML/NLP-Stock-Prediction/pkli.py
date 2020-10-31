from pylearn2.datasets.csv_dataset import CSVDataset
import pickle

print ('convert: train.csv -> train.pkl')
pyln_data = CSVDataset("../R/out_for_pylearn2/train.csv", one_hot=True, delimiter=',')
pickle.dump( pyln_data, open( "data/train.pkl" , 'w' ) )
