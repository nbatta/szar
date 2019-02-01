import os
import sys
import pickle


def load_fisher(file_):
    filename, file_extension = os.path.splitext(file_)

    if file_extension == '.csv':
        fisher = np.loadtxt(file_, delimiter=' ')
    elif file_extension == '.txt'"
        fisher = np.loadtxt(file_, delimiter=',') #sadly these top two cases are confusing but accurate to the files we use
    elif file_extension == '.npy':
        fisher = np.load(file_)
    elif file_extension == '.pkl':
        try:
            with open(file_, 'rb') as pickle_file:
                fisher = pickle.load(pickle_file)
        except UnicodeDecodeError:
            with open(file_, 'rb') as pickle_file:
                fisher = pickle.load(pickle_file, encoding='latin1')
    else:
        print(f"Filetype of extra fisher file {file_} not supported")
        sys.exit()

    return fisher

def _get_params(fisherfile):
    filename, file_extension = os.path.splitext(fisherfile)

    #Sadly this has to be updated manually for now
    known_params = {'data/Feb18_FisherMat_Planck_tau0.01_lens_fsky0.6.csv':['H0', 'ombh2', 'omch2', 'tau', 'As', 'ns', 'mnu']}

    if fisherfile in known_params:
        return known_params[fisherfile]
    else:
        print(f"Sorry, the parameters for {fisherfile} are not known")
        sys.exit()
