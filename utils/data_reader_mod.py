import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import torch
import datetime

def importData(directory, x_range, y_range):
    # pull data into python, should be either for training set or eval set
    train_data_files = []
    for file in os.listdir(os.path.join(directory)):
        if file.endswith('.csv'):
            train_data_files.append(file)
    print(train_data_files)
    # get data
    ftr = []
    lbl = []
    for file_name in train_data_files:
        # import full arrays
        ftr_array = pd.read_csv(os.path.join(directory, file_name), delimiter=',',
                                header=None, usecols=x_range)
        lbl_array = pd.read_csv(os.path.join(directory, file_name), delimiter=',',
                                header=None, usecols=y_range)
        # append each data point to ftr and lbl
        for params, curve in zip(ftr_array.values, lbl_array.values):
            ftr.append(params)
            lbl.append(curve)
    ftr = np.array(ftr, dtype='float32')
    lbl = np.array(lbl, dtype='float32')
    for i in range(len(ftr[0, :])):
        print('For feature {}, the max is {} and min is {}'.format(i, np.max(ftr[:, i]), np.min(ftr[:, i])))
    return ftr, lbl


# check that the data we're using is distributed uniformly and generate some plots
def check_data(input_directory, col_range=range(2, 10), col_names=('h1','h2','h3','h4','r1','r2','r3','r4')):
    for file in os.listdir(input_directory):
        if file.endswith('.csv'):
            print('\n histogram for file {}'.format(os.path.join(input_directory, file)))
            with open(os.path.join(input_directory, file)) as f:
                data = pd.read_csv(f, header=None, delimiter=',', usecols=col_range,
                                   names=col_names)
                for name in col_names:
                    print('{} unique values for {}: {}'.format(len(data[name].unique()),
                                                               name,
                                                               np.sort(data[name].unique()))
                          )
                hist = data.hist(bins=13, figsize=(10, 5))
                plt.tight_layout()
                plt.show()
                print('done plotting column data')




# add columns of derived values to the input data
# for now, just ratios of the inputs
def addColumns(input_directory, output_directory, x_range, y_range):
    print('adding columns...')
    print('importing data')
    data_files = []
    for file in os.listdir(os.path.join(input_directory)):
        if file.endswith('.csv'):
            data_files.append(file)
    for file in data_files:
        ftr = pd.read_csv(os.path.join(input_directory, file), delimiter=',', usecols=x_range,
                          names=['id0', 'id1'] + ['ftr' + str(i) for i in range(8)])
        lbl = pd.read_csv(os.path.join(input_directory, file), delimiter=',', usecols=y_range, header=None)

        print('computing new columns')
        newCol = 0  # count the number of new columns added
        for i in range(2, 6):  # first four are heights
            for j in range(6, 10):  # second four are radii
                ftr['ftr{}'.format(j-2)+'/'+'ftr{}'.format(i-2)] = ftr.apply(lambda row: row.iloc[j]/row.iloc[i], axis=1)
                newCol += 1
        print('total new columns added is {}\n'.format(newCol))
        print('exporting')
        data_total = pd.concat([ftr, pd.DataFrame(lbl)], axis=1)
        # data_total['2010'] = data_total.str.replace('\n', ' ')
        with open(os.path.join(output_directory, file[:-4] + '_div01.csv'), 'a') as file_out:
            # for some stupid reason to_csv seems to insert a blank line between every single data line
            # maybe try to fix this issue later
            #data_total.to_csv(file_out, sep=',', index=False, header=False)
            data_out = data_total.values
            np.savetxt(file_out, data_out, delimiter=',', fmt='%f')
    print('done')

# finds simulation files in input_dir and finds + saves the subset that adhere to the geometry contraints r_bound
# and h_bound
def gridShape(input_dir, output_dir, shapeType, r_bounds, h_bounds):

    files_to_filter = []
    for file in os.listdir(input_dir):
        if file.endswith('.csv'):
            files_to_filter.append(os.path.join(input_dir, file))

    print('filtering through {} files...'.format(len(files_to_filter)))
    print('bounds on radii: [{}, {}], bounds on heights: [{}, {}]...'.format(r_bounds[0], r_bounds[1],
                                                                       h_bounds[0], h_bounds[1]))
    lengthsPreFilter = []
    lengthsPostFilter = []
    for file in files_to_filter:
        with open(file, 'r') as f:
            geom_specs = pd.read_csv(f, delimiter=',', header=None).values
            geoms_filt = []
            geoms_filtComp = []
            lengthsPreFilter.append(len(geom_specs))
            if shapeType=='corner':
                print('cutting a corner of the data...')
                for geom_spec in geom_specs:
                    hs = geom_spec[2:6]
                    rs = geom_spec[6:10]
                    if (np.all(hs >= h_bounds[0]) and np.all(hs <= h_bounds[1])) or \
                       (np.all(rs >= r_bounds[0]) and np.all(rs <= r_bounds[1])):
                        geoms_filt.append(geom_spec)
                    else:
                        geoms_filtComp.append(geom_spec)
            elif shapeType=='rCut':
                print('cutting based on r values only...')
                for geom_spec in geom_specs:
                    rs = geom_spec[6:10]
                    if (np.all(rs >= r_bounds[0]) and np.all(rs <= r_bounds[1])):
                        geoms_filt.append(geom_spec)
                    else:
                        geoms_filtComp.append(geom_spec)
            elif shapeType == 'hCut':
                print('cutting based on h values only...')
                for geom_spec in geom_specs:
                    hs = geom_spec[2:6]
                    if (np.all(hs >= h_bounds[0]) and np.all(hs <= h_bounds[1])):
                        geoms_filt.append(geom_spec)
                    else:
                        geoms_filtComp.append(geom_spec)
            else:
                print('shapeType {} is not valid.'.format(shapeType))
                return
            geoms_filt = np.array(geoms_filt)
            geoms_filtComp = np.array(geoms_filtComp)
            lengthsPostFilter.append(len(geoms_filt))
            print('{} reduced from {} to {}, ({}%)'.format(file, lengthsPreFilter[-1],
                                                           lengthsPostFilter[-1],
                                                           100*np.round(lengthsPostFilter[-1]/lengthsPreFilter[-1], 4)))

        save_file = os.path.join(output_dir, os.path.split(file)[-1][:-4] + '_filt')
        # save the filtered geometries, for training
        with open(save_file + '.csv', 'w+') as f:
            np.savetxt(f, geoms_filt, delimiter=',', fmt='%f')

        # save the all the goemetries filtered out, for evaluation
        with open(save_file + 'Comp.csv', 'w+') as f:
            np.savetxt(f, geoms_filtComp, delimiter=',', fmt='%f')


    print('\nAcross all files: of original {} combos, {} remain ({}%)'.format(sum(lengthsPreFilter),
                                                                              sum(lengthsPostFilter),
                                                                              100*np.round(sum(lengthsPostFilter)/ \
                                                                                       sum(lengthsPreFilter), 4)
                                                                              ))


def read_data_meta_material( x_range, y_range, geoboundary,  batch_size=128,
                 data_dir=os.path.abspath(''), rand_seed=1234, normalize_input = True, test_ratio=0.02,
                             eval_data_all=False):
    """
      :param input_size: input size of the arrays
      :param output_size: output size of the arrays
      :param x_range: columns of input data in the txt file
      :param y_range: columns of output data in the txt file
      :param cross_val: number of cross validation folds
      :param val_fold: which fold to be used for validation
      :param batch_size: size of the batch read every time
      :param shuffle_size: size of the batch when shuffle the dataset
      :param data_dir: parent directory of where the data is stored, by default it's the current directory
      :param rand_seed: random seed
      :param test_ratio: if this is not 0, then split test data from training data at this ratio
                         if this is 0, use the dataIn/eval files to make the test set
      """
    """
    Read feature and label
    :param is_train: the dataset is used for training or not
    :param train_valid_tuple: if it's not none, it will be the names of train and valid files
    :return: feature and label read from csv files, one line each time
    """
    dataset = MetaMaterialDataSet(x_range, y_range, geoboundary)
    test_len = int(test_ratio*len(dataset))
    train_len = len(dataset)-test_len
    train_set, test_set = torch.utils.data.random_split(dataset, [train_len, test_len])
    train_set = permutate_periodicity(train_set)
    print("Length of training set is: "+str(len(train_set)))
    print("Length of test set is: "+str(len(test_set)))
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, num_workers=5)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, num_workers=5)
    return train_loader, test_loader


def get_data_into_loaders(data_x, data_y, batch_size, DataSetClass, rand_seed=1234, test_ratio=0.05):
    """
    Helper function that takes structured data_x and data_y into dataloaders
    :param data_x: the structured x data
    :param data_y: the structured y data
    :param rand_seed: the random seed
    :param test_ratio: The testing ratio
    :return: train_loader, test_loader: The pytorch data loader file
    """
    # Normalize the input
    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=test_ratio,
                                                        random_state=rand_seed)
    print('total number of training sample is {}, the dimension of the feature is {}'.format(len(x_train), len(x_train[0])))
    print('total number of test sample is {}'.format(len(y_test)))

    # Construct the dataset using a outside class
    train_data = DataSetClass(x_train, y_train)
    test_data = DataSetClass(x_test, y_test)

    # Construct train_loader and test_loader
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

    return train_loader, test_loader


def normalize_np(x):
    """
    Normalize the x into [-1, 1] range in each dimension [:, i]
    :param x: np array to be normalized
    :return: normalized np array
    """
    for i in range(len(x[0])):
        x_max = np.max(x[:, i])
        x_min = np.min(x[:, i])
        x_range = (x_max - x_min ) /2.
        x_avg = (x_max + x_min) / 2.
        x[:, i] = (x[:, i] - x_avg) / x_range
        assert np.max(x[:, i]) == 1, 'your normalization is wrong'
        assert np.min(x[:, i]) == -1, 'your normalization is wrong'
    return x


def read_data_sine_test_1d(flags, eval_data_all=False):
    """
    Data reader function for testing sine wave 1d data (1d to 1d)
    :param flags: Input flags
    """
    data_dir = os.path.join(flags.data_dir, 'Simulated_DataSets/Sine_test/')
    data_x = pd.read_csv(data_dir + 'data_x.csv', header=None).astype('float32').values
    data_y = pd.read_csv(data_dir + 'data_y.csv', header=None).astype('float32').values
    if eval_data_all:
        return get_data_into_loaders(data_x, data_y, flags.batch_size, SimulatedDataSet_class_1d_to_1d, test_ratio=0.99)
    return get_data_into_loaders(data_x, data_y, flags.batch_size, SimulatedDataSet_class_1d_to_1d, test_ratio=flags.test_ratio)
  

def read_data_ballistics(flags, eval_data_all=False):
    """
    Data reader function for the ballistic data set
    :param flags: Input flags
    :return train_loader and test_loader in pytorch data set format (unnormalized)
    """
    data_dir = os.path.join(flags.data_dir, 'Simulated_DataSets/Ballistics/')
    data_x = pd.read_csv(data_dir + 'data_x.csv', header=None).astype('float32').values
    data_y = pd.read_csv(data_dir + 'data_y.csv', header=None).astype('float32').values
    if eval_data_all:
        return get_data_into_loaders(data_x, data_y, flags.batch_size, SimulatedDataSet_class, test_ratio=0.999)
    return get_data_into_loaders(data_x, data_y, flags.batch_size, SimulatedDataSet_class, test_ratio=flags.test_ratio)


def read_data_gaussian_mixture(flags, eval_data_all=False):
    """
    Data reader function for the gaussian mixture data set
    :param flags: Input flags
    :return: train_loader and test_loader in pytorch data set format (normalized)
    """
    # Read the data
    data_dir = os.path.join(flags.data_dir, 'Simulated_DataSets/Gaussian_Mixture/')
    data_x = pd.read_csv(data_dir + 'data_x.csv', header=None).astype('float32').values
    data_y = pd.read_csv(data_dir + 'data_y.csv', header=None).astype('float32').values
    data_y = np.squeeze(data_y)                             # Squeeze since this is a 1 column label
    data_x = normalize_np(data_x)
    if eval_data_all:
        return get_data_into_loaders(data_x, data_y, flags.batch_size, SimulatedDataSet_class, test_ratio=0.999)
    return get_data_into_loaders(data_x, data_y, flags.batch_size, SimulatedDataSet_class, test_ratio=flags.test_ratio)


def read_data_sine_wave(flags, eval_data_all=False):
    """
    Data reader function for the sine function data set
    :param flags: Input flags
    :return: train_loader and test_loader in pytorch data set format (normalized)
    """
    data_dir = os.path.join(flags.data_dir, 'Simulated_DataSets/Sinusoidal_Wave/')
    data_x = pd.read_csv(data_dir + 'data_x.csv', header=None).astype('float32').values
    data_y = pd.read_csv(data_dir + 'data_y.csv', header=None).astype('float32').values
    #data_x = normalize_np(data_x)
    #data_y = normalize_np(data_y)
    if eval_data_all:
        return get_data_into_loaders(data_x, data_y, flags.batch_size, SimulatedDataSet_class, test_ratio=0.999)
    return get_data_into_loaders(data_x, data_y, flags.batch_size, SimulatedDataSet_class, test_ratio=flags.test_ratio)


def read_data_naval_propulsion(flags, eval_data_all=False):
    """
    Data reader function for the naval propulsion data set
    :param flags: Input flags
    :return: train_loader and test_loader in pytorch data set format (normalized)
    """
    data_dir = os.path.join(flags.data_dir, 'Simulated_DataSets/Naval_Propulsion/')
    data_x = pd.read_csv(data_dir + 'data_x.csv', header=None).astype('float32').values
    data_y = pd.read_csv(data_dir + 'data_y.csv', header=None).astype('float32').values
    data_x = normalize_np(data_x)
    data_y = normalize_np(data_y)
    if eval_data_all:
        return get_data_into_loaders(data_x, data_y, flags.batch_size, SimulatedDataSet_regress, test_ratio=0.999)
    return get_data_into_loaders(data_x, data_y, flags.batch_size, SimulatedDataSet_regress, test_ratio=flags.test_ratio)


def read_data_robotic_arm(flags, eval_data_all=False):
    """
    Data reader function for the robotic arm data set
    :param flags: Input flags
    :return: train_loader and test_loader in pytorch data set format (normalized)
    """
    data_dir = os.path.join(flags.data_dir, 'Simulated_DataSets/Robotic_Arm/')
    data_x = pd.read_csv(data_dir + 'data_x.csv', header=None).astype('float32').values
    data_y = pd.read_csv(data_dir + 'data_y.csv', header=None).astype('float32').values
    if eval_data_all:
        return get_data_into_loaders(data_x, data_y, flags.batch_size, SimulatedDataSet_regress, test_ratio=0.999)
    return get_data_into_loaders(data_x, data_y, flags.batch_size, SimulatedDataSet_regress, test_ratio=flags.test_ratio)


def read_data(flags, eval_data_all=False):
    """
    The data reader allocator function
    The input is categorized into couple of different possibilities
    0. meta_material
    1. gaussian_mixture
    2. sine_wave
    3. naval_propulsion
    4. robotic_arm
    5. ballistics
    :param flags: The input flag of the input data set
    :param eval_data_all: The switch to turn on if you want to put all data in evaluation data
    :return:
    """
    if flags.data_set == 'meta_material':
        train_loader, test_loader = read_data_meta_material(x_range=flags.x_range,
                                                            y_range=flags.y_range,
                                                            geoboundary=flags.geoboundary,
                                                            batch_size=flags.batch_size,
                                                            normalize_input=flags.normalize_input,
                                                            data_dir=flags.data_dir,
                                                            eval_data_all=eval_data_all,
                                                            test_ratio=flags.test_ratio)
        # Reset the boundary is normalized
        if flags.normalize_input:
            flags.geoboundary_norm = [-1, 1, -1, 1]
    elif flags.data_set == 'gaussian_mixture':
        train_loader, test_loader = read_data_gaussian_mixture(flags, eval_data_all=eval_data_all)
    elif flags.data_set == 'sine_wave':
        train_loader, test_loader = read_data_sine_wave(flags, eval_data_all=eval_data_all)
    elif flags.data_set == 'naval_propulsion':
        train_loader, test_loader = read_data_naval_propulsion(flags, eval_data_all=eval_data_all)
    elif flags.data_set == 'robotic_arm':
        train_loader, test_loader = read_data_robotic_arm(flags, eval_data_all=eval_data_all)
    elif flags.data_set == 'ballistics':
        train_loader, test_loader = read_data_ballistics(flags,  eval_data_all=eval_data_all)
    elif flags.data_set == 'sine_test_1d':
        train_loader, test_loader = read_data_sine_test_1d(flags, eval_data_all=eval_data_all)
    else:
        sys.exit("Your flags.data_set entry is not correct, check again!")
    return train_loader, test_loader

class MetaMaterialDataSet(Dataset):
    """ The Meta Material Dataset Class """
    def __init__(self, ftr, lbl, bool_train):
        """
        Instantiate the Dataset Object
        :param ftr: the features which is always the Geometry !!
        :param lbl: the labels, which is always the Spectra !!
        :param bool_train:
        """
        self.ftr = ftr
        self.lbl = lbl
        self.bool_train = bool_train
        self.len = len(ftr)

    def __len__(self):
        return self.len

    def __getitem__(self, ind):
        return self.ftr[ind, :], self.lbl[ind, :]


class SimulatedDataSet_class_1d_to_1d(Dataset):
    """ The simulated Dataset Class for classification purposes"""
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.len = len(x)

    def __len__(self):
        return self.len

    def __getitem__(self, ind):
        return self.x[ind], self.y[ind]


class SimulatedDataSet_class(Dataset):
    """ The simulated Dataset Class for classification purposes"""
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.len = len(x)

    def __len__(self):
        return self.len

    def __getitem__(self, ind):
        return self.x[ind, :], self.y[ind]


class SimulatedDataSet_regress(Dataset):
    """ The simulated Dataset Class for regression purposes"""
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.len = len(x)

    def __len__(self):
        return self.len

    def __getitem__(self, ind):
        return self.x[ind, :], self.y[ind, :]

class MetaMaterialUpdatedDataSet(Dataset):
    """ The Meta Material Dataset Class """
    def __init__(self, x_range, y_range, geoboundary, data_dir=os.path.abspath(''), normalize_input = True):
        """
        Instantiate the Dataset Object
        :param ftr: the features which is always the Geometry !!
        :param lbl: the labels, which is always the Spectra !!
        :param bool_train:
        """

        self.xrange = x_range
        self.yrange = y_range
        self.geoboundary = geoboundary
        self.data_dir = data_dir
        self.normalize_input = normalize_input
        self.ftr = self.importData(self.xrange)
        begin_time = datetime.datetime.now()
        self.lbl = self.importData(self.yrange)
        print(datetime.datetime.now() - begin_time)
        self.len = len(self.ftr)
        self.helper = pd.read_csv('D:/Yang_MM_Absorber_ML/helper.csv', delimiter=' ', header=None)


    def __len__(self):
        return self.len

    def __getitem__(self, ind):
        begin_time = datetime.datetime.now()
        #spectra = self.readspectra(int(self.helper.iloc[ind][15]), int(self.helper.iloc[ind][14]))
        spectra = self.lbl[ind, :]
        print(datetime.datetime.now() - begin_time)
        if len(spectra) > 2000:  # For Omar data set
            spectra = spectra[len(spectra) - 2000::1]

        return self.ftr[ind, :], spectra

    def importData(self, drange):
        # pull data into python, should be either for training set or eval set
        train_data_files = []
        for file in os.listdir(os.path.join(self.data_dir, 'dataIn')):
            if file.endswith('.csv'):
                train_data_files.append(file)
        print(train_data_files)
        # get data
        ftr = []
        for file_name in train_data_files:
            # import full arrays
            ftr_array = pd.read_csv(os.path.join(self.data_dir, 'dataIn', file_name), delimiter=' ',
                                    header=None, usecols=drange)
            # append each data point to ftr and lbl
            for params in zip(ftr_array.values):
                ftr.append(params)
        ftr = np.array(ftr, dtype='float32')
        return ftr

    def readspectra(self, findex, il):
        ftr = []
        #print("Currently reading: "+str(os.path.join(self.data_dir, 'dataIn', 'train' + str(findex) + '_augmented.csv')))
        ftr_array = pd.read_csv(os.path.join(self.data_dir, 'dataIn', 'train' + str(findex) + '_augmented.csv'), delimiter=' ',
                                    header=None, usecols=self.yrange)
        # append each data point to ftr and lbl
        ftr = ftr_array.iloc[il]
        ftr = np.array(ftr, dtype='float32')
        return ftr

def permutate_periodicity(geometry_in, spectra_in):
    """
    :param: geometry_in: numpy array of geometry [n x 14] dim
    :param: spectra_in: spectra of the geometry_in [n x k] dim
    :return: output of the augmented geometry, spectra [4n x 14], [4n x k]
    """
    # Get the dimension parameters
    (n, k) = np.shape(spectra_in)
    # Initialize the output
    spectra_out = np.zeros([4 * n, k])
    geometry_out = np.zeros([4 * n, 14])

    #################################################
    # start permutation of geometry (case: 1 - 0123)#
    #################################################
    # case:2 -- 1032
    geometry_c2 = geometry_in[:, [0, 1, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12]]
    # case:3 -- 2301
    geometry_c3 = geometry_in[:, [0, 1, 4, 5, 2, 3, 8, 9, 6, 7, 12, 13, 10, 11]]
    # case:4 -- 3210
    geometry_c4 = geometry_in[:, [0, 1, 5, 4, 3, 2, 9, 8, 7, 6, 13, 12, 11, 10]]

    geometry_out[0 * n:1 * n, :] = geometry_in
    geometry_out[1 * n:2 * n, :] = geometry_c2
    geometry_out[2 * n:3 * n, :] = geometry_c3
    geometry_out[3 * n:4 * n, :] = geometry_c4
    print(n)

    for i in range(4):
        spectra_out[i * n:(i + 1) * n, :] = spectra_in
    return geometry_out.astype('float32'), spectra_out.astype('float32')