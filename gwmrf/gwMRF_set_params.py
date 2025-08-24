import os
from scipy.io import loadmat
from pathlib import Path
import torch

def _check_ischar(value):
    """
    Helper function to replicate `check_ischar(x)` from MATLAB.
    If `value` is a string that can be interpreted as a number, convert it.
    Otherwise, leave it unchanged.
    """
    if isinstance(value, str):
        try:
            if '.' in value or 'e' in value.lower():
                return float(value)
            else:
                return int(value)
        except ValueError:
            return value
    return value

def CBIG_gwMRF_set_prams(**kwargs):
    """
    Python translation of the MATLAB function CBIG_gwMRF_set_prams.m

    Usage:
        prams = CBIG_gwMRF_set_prams(
            start_index=5982,
            runs=10,
            cluster=160,
            smoothcost=1000,
            datacost=0.1,
            ...
        )

    If a parameter is not provided, its default is used (from the original
    MATLAB defaults). Additional logic sets up gradient-prior files, checks
    file existence, etc. The result is a dictionary `prams`.

    Returns:
    --------
    prams : dict
        A dictionary holding all parameters. For debugging, it also stores a
        fileID (Python file handle) in prams['fileID'] if you need logging.
    """

    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    defaultseed = 5392
    defaultk = 20
    default_left_k = 0
    default_right_k = 0
    defaultV = 5000   # smoothcost
    defaultU = 1      # datacost
    defaultruns = 1
    defaultexpo = 15  # exponent in the gradient prior
    defaultiterations = 1000
    default_iter_reduce_gamma = 300
    defaultrandom_init = True
    defaultgrad_prior = 'gordon'
    defaultfsaverage = 'fsaverage6'
    defaultpotts = False
    
    
    # # Time Series data for lh and rh, This should be fixed for location after processing 
    import nibabel as nib
    default_lh_avg_file = os.path.join(current_dir, '..', '..', 'results', 'mult_mat', 'lh_mult_matrix.pt')
    default_rh_avg_file = os.path.join(current_dir, '..', '..', 'results', 'mult_mat', 'rh_mult_matrix.pt')
    torch.load(default_lh_avg_file)
    default_initialization_prior = './input/prior_initializations_all.mat'
    default_initialization_motor = './input/clustering/motor_labels.mat'
    default_watershed_files = './input/water_files.mat'
    default_output_folder = './output/'
    default_separate_hemispheres = 1
    default_local_concentration = 0.0
    default_estimate_gamma = 1
    default_reduce_gamma = 1
    default_graphCutIterations = 1000
    default_first_gamma = 50000
    default_start_gamma = 2 * default_first_gamma
    default_kappa_vector = 1
    default_data_normalization = 0
    default_left_skip = 0
    default_right_skip = 0
    default_reduce_speed = 5
    default_pca = 1
    default_dim = 1200
    default_alpha = 0.5

    prams = {
        "start_index":         defaultseed,
        "cluster":             defaultk,
        "smoothcost":          defaultV,
        "datacost":            defaultU,
        "runs":                defaultruns,
        "exponential":         defaultexpo,
        "iterations":          defaultiterations,
        "iter_reduce_gamma":   default_iter_reduce_gamma,
        "random_init":         defaultrandom_init,
        "grad_prior":          defaultgrad_prior,
        "fsaverage":           defaultfsaverage,
        "potts":               defaultpotts,
        "lh_avg_file":         default_lh_avg_file,
        "rh_avg_file":         default_rh_avg_file,
        "initialization_prior": default_initialization_prior,
        "initialization_motor": default_initialization_motor,
        "watershed_files":     default_watershed_files,
        "output_folder":       default_output_folder,
        "separate_hemispheres": default_separate_hemispheres,
        "local_concentration": default_local_concentration,
        "left_cluster":        default_left_k,
        "right_cluster":       default_right_k,
        "estimate_gamma":      default_estimate_gamma,
        "graphCutIterations":  default_graphCutIterations,
        "first_gamma":         default_first_gamma,
        "start_gamma":         default_start_gamma,
        "kappa_vector":        default_kappa_vector,
        "data_normalization":  default_data_normalization,
        "reduce_gamma":        default_reduce_gamma,
        "skip_left":           default_left_skip,
        "skip_right":          default_right_skip,
        "reduce_speed":        default_reduce_speed,
        "pca":                 default_pca,
        "dim":                 default_dim,
        "alpha":               default_alpha,
    }


    for key, val in kwargs.items():
        if key in prams:
            prams[key] = val
        else:
            prams[key] = val

    for key in list(prams.keys()):
        prams[key] = _check_ischar(prams[key])


    if prams["local_concentration"] > 0.0:
        default_output_name = (
            f"Graph_Cut_faster__grad_prior_{prams['grad_prior']}_cluster_{prams['cluster']}_"
            f"datacost_{prams['datacost']}_smoothcost_{prams['smoothcost']}_iterations_{prams['iterations']}_"
            f"local_concentration_{prams['local_concentration']}"
        )
    else:
        default_output_name = (
            f"Graph_Cut_faster__grad_prior_{prams['grad_prior']}_cluster_{prams['cluster']}_"
            f"datacost_{prams['datacost']}_smoothcost_{prams['smoothcost']}_iterations_{prams['iterations']}"
        )

    if "output_name" not in prams:
        prams["output_name"] = default_output_name


    if not os.path.exists(prams["output_folder"]):
        os.makedirs(prams["output_folder"])

    print_path = os.path.join(prams["output_folder"], "print")
    if not os.path.exists(print_path):
        os.makedirs(print_path)

    log_filename = (
        f"print_{prams['output_name']}_seed_{prams['start_index']}_to_seed_"
        f"{prams['start_index'] + prams['runs'] - 1}.txt"
    )
    log_filepath = os.path.join(print_path, log_filename)
    fileID = open(log_filepath, "w", encoding="utf-8")
    prams["fileID"] = fileID 

    fileID.write(f"{os.path.join(print_path, log_filename)}\n")

    if prams["grad_prior"] == "midthickness":
        fileID.write("using midthickness gradient prior\n")
        default_lh_grad_file = os.path.join(current_dir, 'input', '3_smooth_lh_borders_744_midthickness_subjects_3_postsmoothing_6.mat')  # "code/lib/input/3_smooth_lh_borders_744_midthickness_subjects_3_postsmoothing_6.mat"
        default_rh_grad_file = os.path.join(current_dir, 'input', '3_smooth_rh_borders_744_midthickness_subjects_3_postsmoothing_6.mat')  # "code/lib/input/3_smooth_rh_borders_744_midthickness_subjects_3_postsmoothing_6.mat"
    elif prams["grad_prior"] == "smoothwm":
        fileID.write("using smootwhm gradient prior\n")
        default_lh_grad_file = os.path.join(current_dir, 'input', '3_smooth_lh_borders_744_smoothwm_subjects_3_postsmoothing_6.mat')   # "code/lib/input/3_smooth_lh_borders_744_smoothwm_subjects_3_postsmoothing_6.mat"
        default_rh_grad_file = os.path.join(current_dir, 'input', '3_smooth_rh_borders_744_smoothwm_subjects_3_postsmoothing_6.mat')   # "code/lib/input/3_smooth_rh_borders_744_smoothwm_subjects_3_postsmoothing_6.mat"
    elif prams["grad_prior"] == "combined":
        fileID.write("using combined gradient prior\n")
        default_lh_grad_file = os.path.join(current_dir, 'input', '3_smooth_lh_borders_744_combined_subjects_3_postsmoothing_6.mat')   # "code/lib/input/3_smooth_lh_borders_744_combined_subjects_3_postsmoothing_6.mat"
        default_rh_grad_file = os.path.join(current_dir, 'input', '3_smooth_rh_borders_744_combined_subjects_3_postsmoothing_6.mat')   #" code/lib/input/3_smooth_rh_borders_744_combined_subjects_3_postsmoothing_6.mat"
    elif prams["grad_prior"] == "gordon":
        fileID.write("using gordon gradient prior\n")
        default_lh_grad_file = os.path.join(current_dir, 'input', '3_smooth_lh_borders_120_gordon_subjects_3_postsmoothing_6.mat') # "code/lib/input/3_smooth_lh_borders_120_gordon_subjects_3_postsmoothing_6.mat"
        default_rh_grad_file = os.path.join(current_dir, 'input', '3_smooth_rh_borders_120_gordon_subjects_3_postsmoothing_6.mat') # "code/lib/input/3_smooth_rh_borders_120_gordon_subjects_3_postsmoothing_6.mat"
    elif prams["grad_prior"] == "gordon_min":
        fileID.write("using gordon gradient prior\n")
        default_lh_grad_file = os.path.join(current_dir, 'input', '3_smooth_lh_borders_120_gordon_subjects_3_postsmoothing_6_min.mat') # "code/lib/input/3_smooth_lh_borders_120_gordon_subjects_3_postsmoothing_6_min.mat"
        default_rh_grad_file = os.path.join(current_dir, 'input', '3_smooth_rh_borders_120_gordon_subjects_3_postsmoothing_6_min.mat') #"code/lib/input/3_smooth_rh_borders_120_gordon_subjects_3_postsmoothing_6_min.mat"
    elif prams["grad_prior"] == "gordon_fs5":
        fileID.write("using gordon gradient prior\n")
        default_lh_grad_file = os.path.join(current_dir, 'input', '3_smooth_lh_borders_120_gordon_subjects_3_postsmoothing_6_fs5.mat')  # "code/lib/input/3_smooth_lh_borders_120_gordon_subjects_3_postsmoothing_6_fs5.mat"
        default_rh_grad_file = os.path.join(current_dir, 'input', '3_smooth_rh_borders_120_gordon_subjects_3_postsmoothing_6_fs5.mat')  # "code/lib/input/3_smooth_rh_borders_120_gordon_subjects_3_postsmoothing_6_fs5.mat"
    elif prams["grad_prior"] == "gordon_water":
        fileID.write("using gordon water gradient prior\n")
        default_lh_grad_file = os.path.join(current_dir, 'input', '3_smooth_lh_borders_120_gordon_subjects_3_postsmoothing_6_water.mat')  # "code/lib/input/3_smooth_lh_borders_120_gordon_subjects_3_postsmoothing_6_water.mat"
        default_rh_grad_file = os.path.join(current_dir, 'input', '3_smooth_rh_borders_120_gordon_subjects_3_postsmoothing_6_water.mat')  # "code/lib/input/3_smooth_rh_borders_120_gordon_subjects_3_postsmoothing_6_water.mat"
    else:
        fileID.write("Warning: unknown gradient prior, will use gordon\n")
        default_lh_grad_file = os.path.join(current_dir, 'input', '3_smooth_lh_borders_120_gordon_subjects_3_postsmoothing_6.mat') # "code/lib/input/3_smooth_lh_borders_120_gordon_subjects_3_postsmoothing_6.mat"
        default_rh_grad_file = os.path.join(current_dir, 'input', '3_smooth_rh_borders_120_gordon_subjects_3_postsmoothing_6.mat') # "code/lib/input/3_smooth_rh_borders_120_gordon_subjects_3_postsmoothing_6.mat"

    if "lh_grad_file" not in prams:
        prams["lh_grad_file"] = default_lh_grad_file
    if "rh_grad_file" not in prams:
        prams["rh_grad_file"] = default_rh_grad_file


    fileID.write(f"outputname: {prams['output_name']}\n")
    fileID.write(
        f"using profile {prams['lh_avg_file']} and {prams['rh_avg_file']}\n"
    )

    if prams["start_index"] == defaultseed:
        fileID.write("WARNING: You use the default seed, which makes sense only for debugging\n")

    if not os.path.isfile(prams["lh_grad_file"]):
        raise FileNotFoundError(f"inputfile {prams['lh_grad_file']} does not exist.")
    if not os.path.isfile(prams["rh_grad_file"]):
        raise FileNotFoundError(f"inputfile {prams['rh_grad_file']} does not exist.")
    
    
    if not os.path.isfile(prams["lh_avg_file"]):
        raise FileNotFoundError(f"inputfile {prams['lh_avg_file']} does not exist.")
    if not os.path.isfile(prams["rh_avg_file"]):
        raise FileNotFoundError(f"inputfile {prams['rh_avg_file']} does not exist.")

    if prams["left_cluster"] == 0:
        prams["left_cluster"] = prams["cluster"]
        prams["right_cluster"] = prams["cluster"]

    if prams["local_concentration"] < 0:
        raise ValueError("local concentration should be non-negative")

    return prams


