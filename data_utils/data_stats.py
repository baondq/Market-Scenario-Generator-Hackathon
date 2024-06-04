import numpy as np
import pandas as pd
import pickle


def remove_outliers(data, need_lower=True):
    q1 = np.quantile(data, 0.25, axis=(0,1))
    q3 = np.quantile(data, 0.75, axis=(0,1))
    upper = q3 + 1.5 * (q3 - q1)
    upper = np.broadcast_to(np.expand_dims(upper, axis=(0,1)), shape=data.shape)
    if need_lower:
        lower = q1 - 1.5 * (q3 - q1)
        lower = np.broadcast_to(np.expand_dims(lower, axis=(0,1)), shape=data.shape)
        data = np.where((data >= lower) & (data <= upper), data, np.NaN)
    else:
        data = np.where(data <= upper, data, np.NaN)
    return data


def compute_stats(regular_data, crisis_data, out_file):
    # get log and volatility of each data
    log_reg_data = regular_data[:,:,::2]
    vol_reg_data = regular_data[:,:,1::2]
    log_cri_data = crisis_data[:,:,::2]
    vol_cri_data = crisis_data[:,:,1::2]
    # compute stats for regular log
    log_reg_data = remove_outliers(log_reg_data, need_lower=True)
    log_reg_mean = np.nanmean(log_reg_data, axis=(0,1))
    log_reg_std = np.nanstd(log_reg_data, axis=(0,1))
    print(log_reg_mean, log_reg_std)
    # compute stats for regular volatility
    vol_reg_data = remove_outliers(vol_reg_data, need_lower=False)
    vol_reg_mean = np.nanmean(vol_reg_data, axis=(0,1))
    vol_reg_std = np.nanstd(vol_reg_data, axis=(0,1))
    print(vol_reg_mean, vol_reg_std)
    # compute stats for crisis log
    log_cri_data = remove_outliers(log_cri_data, need_lower=True)
    log_cri_mean = np.nanmean(log_cri_data, axis=(0,1))
    log_cri_std = np.nanstd(log_cri_data, axis=(0,1))
    print(log_cri_mean, log_cri_std)
    # compute stats for crisis volatility
    vol_cri_data = remove_outliers(vol_cri_data, need_lower=False)
    vol_cri_mean = np.nanmean(vol_cri_data, axis=(0,1))
    vol_cri_std = np.nanstd(vol_cri_data, axis=(0,1))
    print(vol_cri_mean, vol_cri_std)
    # compute cov and corr
    reg_df = pd.DataFrame(np.concatenate((log_reg_data, vol_reg_data), axis=-1).reshape((-1,regular_data.shape[-1])))
    reg_cov = reg_df.cov().values
    reg_corr = reg_df.corr().values
    cri_df = pd.DataFrame(np.concatenate((log_cri_data, vol_cri_data), axis=-1).reshape((-1,crisis_data.shape[-1])))
    cri_cov = cri_df.cov().values
    cri_corr = cri_df.corr().values
    # store stats
    stats = {
        "log_reg_mean": log_reg_mean,
        "log_reg_std": log_reg_std,
        "vol_reg_mean": vol_reg_mean,
        "vol_reg_std": vol_reg_std,
        "log_cri_mean": log_cri_mean,
        "log_cri_std": log_cri_std,
        "vol_cri_mean": vol_cri_mean,
        "vol_cri_std": vol_cri_std,
        "reg_cov": reg_cov,
        "reg_corr": reg_corr,
        "cri_cov": cri_cov,
        "cri_corr": cri_corr
    }
    with open(out_file, "wb") as f:
        pickle.dump(stats, f)
    


if __name__ == "__main__":

    # with open("data/ref_data.pkl", "rb") as f:
    #     ref_data = pickle.load(f)

    # with open("data/ref_label.pkl", "rb") as f:
    #     ref_label = pickle.load(f)
    
    # regular_data = ref_data[ref_label[:,0]==0]
    # crisis_data = ref_data[ref_label[:,0]==1]
    # compute_stats(regular_data, crisis_data, "data/detail_stats.pkl")

    with open("data/class_stats.pkl", "rb") as f:
        old_stats = pickle.load(f)
    
    stats = {
        "log_reg_mean": log_reg_mean,
        "log_reg_std": log_reg_std,
        "vol_reg_mean": vol_reg_mean,
        "vol_reg_std": vol_reg_std,
        "log_cri_mean": log_cri_mean,
        "log_cri_std": log_cri_std,
        "vol_cri_mean": vol_cri_mean,
        "vol_cri_std": vol_cri_std,
        "reg_cov": reg_cov,
        "reg_corr": reg_corr,
        "cri_cov": cri_cov,
        "cri_corr": cri_corr
    }
