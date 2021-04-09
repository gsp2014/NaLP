import tensorflow as tf
import numpy as np

def replace_val(n_values, last_idx, role_val, arity, new_facts_indexes, new_facts_values, whole_train_facts):
    """
    Replace values randomly to get negative samples
    """
    role_ind = (np.random.randint(np.iinfo(np.int32).max) % arity) * 2
    tmp_role = new_facts_indexes[last_idx, role_ind]
    tmp_len = len(role_val[tmp_role])
    rdm_w = np.random.randint(0, tmp_len)  # [low,high)

    # Sample a random value
    times = 1
    tmp_array = new_facts_indexes[last_idx]
    tmp_array[role_ind+1] = role_val[tmp_role][rdm_w]
    while (tuple(tmp_array) in whole_train_facts):
        if (tmp_len == 1) or (times > 2*tmp_len) or (times > 100):
            tmp_array[role_ind+1] = np.random.randint(0, n_values)
        else:
            rdm_w = np.random.randint(0, tmp_len)
            tmp_array[role_ind+1] = role_val[tmp_role][rdm_w]
        times = times + 1
    new_facts_indexes[last_idx, role_ind+1] = tmp_array[role_ind+1]
    new_facts_values[last_idx] = [-1]

def replace_role(n_roles, last_idx, arity, new_facts_indexes, new_facts_values, whole_train_facts):
    """
    Replace roles randomly to get negative samples
    """
    role_ind = (np.random.randint(np.iinfo(np.int32).max) % arity) * 2
    # Sample a random role
    tmp_array = new_facts_indexes[last_idx]
    tmp_array[role_ind] = np.random.randint(0, n_roles)
    while (tuple(tmp_array) in whole_train_facts):
        tmp_array[role_ind] = np.random.randint(0, n_roles)
    new_facts_indexes[last_idx, role_ind] = tmp_array[role_ind]
    new_facts_values[last_idx] = [-1]

def replace_nrv(train_rvs, rnum, last_idx, arity, new_facts_indexes, new_facts_values, whole_train_facts):
    rdm_num = np.random.randint(np.iinfo(np.int32).max) % (arity-1) + 1
    rdm_inds = np.random.randint(0, np.iinfo(np.int32).max, rdm_num) % arity
    tmp_array = new_facts_indexes[last_idx]
    rdm_ws = np.random.randint(0, len(train_rvs), len(rdm_inds))
    for i in range(len(rdm_inds)):
        tmp_array[2*rdm_inds[i]] = train_rvs[rdm_ws[i]][0]
        tmp_array[2*rdm_inds[i]+1] = train_rvs[rdm_ws[i]][1]
    while (tuple(tmp_array) in whole_train_facts):
        rdm_ws = np.random.randint(0, len(train_rvs), len(rdm_inds))
        for i in range(len(rdm_inds)):
            tmp_array[2*rdm_inds[i]] = train_rvs[rdm_ws[i]][0]
            tmp_array[2*rdm_inds[i]+1] = train_rvs[rdm_ws[i]][1]
    new_facts_indexes[last_idx] = tmp_array
    new_facts_values[last_idx] = [-1]

def Batch_Loader(train_batch_indexes, train_batch_values, train_rvs, values_indexes, roles_indexes, role_val, batch_size, arity, whole_train_facts):
    new_facts_indexes = np.empty((batch_size*2, 2*arity)).astype(np.int32)
    new_facts_values = np.empty((batch_size*2, 1)).astype(np.float32)

    idxs = np.random.randint(0, len(train_batch_values), batch_size)
    new_facts_indexes[:batch_size, :] = train_batch_indexes[idxs, :]
    new_facts_values[:batch_size] = train_batch_values[idxs, :]

    # Copy everyting in advance
    new_facts_indexes[batch_size:(batch_size*2), :] = np.tile(
        new_facts_indexes[:batch_size, :], (1, 1))
    new_facts_values[batch_size:(batch_size*2)] = np.tile(
        new_facts_values[:batch_size], (1, 1))
    n_values = len(values_indexes)
    n_roles = len(roles_indexes)
    for cur_idx in range(batch_size):
        if np.random.randint(np.iinfo(np.int32).max) % 2 == 0:  # replace one role or value
            val_role = np.random.randint(np.iinfo(np.int32).max) % (n_values+n_roles)
            if val_role < n_values:  # 0~(n_values-1)
                replace_val(n_values, batch_size+cur_idx, role_val, arity, new_facts_indexes, new_facts_values, whole_train_facts)
            else:
                replace_role(n_roles, batch_size+cur_idx, arity, new_facts_indexes, new_facts_values, whole_train_facts)
        else:
            rnum = np.random.randint(np.iinfo(np.int32).max) % arity
            replace_nrv(train_rvs, rnum, batch_size+cur_idx, arity, new_facts_indexes, new_facts_values, whole_train_facts)

    return new_facts_indexes[:batch_size*2, :], new_facts_values[:batch_size*2]
