#!/usr/bin/python

import numpy as np
import pickle
import chumpy
import sys
from scipy.sparse import issparse

def load_data(data):
    v_template = np.array(data['v_template'])
    Jreg = data['J_regressor']

    if issparse(Jreg):
        Jreg = Jreg.todense()

    shapedirs = np.array(data['shapedirs'])
    parent = np.array(data['kintree_table'][0, :])

    RelJreg = np.vstack([Jreg[i, :] - Jreg[parent[i], :] for i in
                         range(1, Jreg.shape[0])])
    RelJreg = np.vstack((Jreg[0, :], RelJreg))

    JDirs = np.stack([Jreg.dot(shapedirs[:, :, i]).ravel()
                      for i in range(shapedirs.shape[2])])
    J = Jreg.dot(v_template).ravel()

    RelJDirs = np.stack([RelJreg.dot(shapedirs[:, :, i]).ravel()
                         for i in range(shapedirs.shape[2])])
    RelJ = RelJreg.dot(v_template).ravel()

    vDirs = np.stack([shapedirs[:, :, i].ravel()
                      for i in range(shapedirs.shape[2])])
    v = v_template.ravel()

    return {'JDirs': JDirs, 'J': J, 'RelJDirs': RelJDirs, 'RelJ': RelJ, 'vDirs': vDirs, 'v': v}


def load(file):
    with open(file,'rb') as f:
        data = pickle.load(f,encoding = 'latin1')
        
    return load_data(data)


smpl = load(sys.argv[2])
np.savez(sys.argv[1],**smpl)