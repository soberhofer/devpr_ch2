import torch
import torch.nn as nn
import subprocess
import pandas as pd
import numpy as np
import os
import hydra
from omegaconf import DictConfig, OmegaConf
import hydra.utils as hyu

from dataset.dataset_ESC50 import ESC50, InMemoryESC50, download_extract_zip # Added InMemoryESC50
from train_crossval import test, make_model, global_stats
# import config # Removed config import


@hydra.main(config_path="conf", config_name="config", version_base=None) # Corrected config_path
def main(cfg: DictConfig):
    # Print the configuration - useful for debugging
    print(OmegaConf.to_yaml(cfg))

    reproducible = False # This could be moved to Hydra config cfg.testing.reproducible
    data_path = hyu.to_absolute_path(cfg.data.path) # Use Hydra config and resolve path
    
    # Determine device
    use_cuda = torch.cuda.is_available()
    use_mps = torch.backends.mps.is_available()
    if use_cuda:
        device = torch.device(f"cuda:{cfg.testing.device_id}") # Use Hydra config
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    check_data_reproducibility = False
    if reproducible:
        # improve reproducibility
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(0)
        # for debugging only, uncomment
        #check_data_reproducibility = True

    # digits for logging
    float_fmt = ".3f"
    pd.options.display.float_format = (f'{{:,{float_fmt}}}').format # Python 3.6+ f-string

    # experiment_root is the path to the trained model's output directory
    # This should come from cfg.testing.cvpath and be an absolute path or resolvable
    experiment_root = hyu.to_absolute_path(cfg.testing.cvpath) # Use Hydra config
    
    # The download logic might need to be conditional based on whether experiment_root exists
    # or if a specific "download_if_missing" flag is set in config.
    if not os.path.isdir(experiment_root) and cfg.testing.get("download_sample_run_if_missing", False): # Make download conditional
        print(f'Trained model directory not found at {experiment_root}. Downloading sample run...')
        download_extract_zip(
            url=cfg.testing.get("sample_run_url", 'https://cloud.technikum-wien.at/s/PiHsFtnB69cqxPE/download/sample-run.zip'), # Make URL configurable
            file_path=experiment_root + '.zip', # This assumes cvpath is a dir name, not a zip file path
        )
    elif not os.path.isdir(experiment_root):
        print(f"Error: Trained model directory not found at {experiment_root}")
        print("Please specify a valid 'testing.cvpath' in your Hydra configuration or override it.")
        print("Example: python test_crossval.py testing.cvpath=/path/to/your/training_output_folder")
        return # Exit if path is invalid and not downloading

    # instantiate model
    print('*****')
    print("WARNING: Using hardcoded global mean and std. Depends on feature settings!")
    # Model instantiation will be changed to make_model(cfg)
    model = make_model(cfg) # Changed to pass full cfg
    model = model.to(device)
    print('*****')

    criterion = nn.CrossEntropyLoss().to(device)

    # for all folds
    scores = {}
    # Initialize probs dictionary based on checkpoints from Hydra config
    probs = {model_file_name: {} for model_file_name in cfg.testing.checkpoints}
    
    for test_fold in cfg.data.test_folds: # Use Hydra config
        # experiment here refers to the specific fold directory within experiment_root
        experiment_fold_path = os.path.join(experiment_root, str(test_fold))
        if not os.path.isdir(experiment_fold_path):
            print(f"Warning: Fold directory {experiment_fold_path} not found in {experiment_root}. Skipping fold {test_fold}.")
            continue

        # Data loading logic will be updated here later to use cfg.data.dataset_type
        # For now, keep original ESC50 but use cfg for batch_size
        # This section will be replaced in Phase 2
        dataset_class_to_use = None
        if cfg.data.dataset_type == "in_memory":
            dataset_class_to_use = InMemoryESC50
            print(f"Fold {test_fold}: Using InMemoryESC50 dataset for testing.")
        elif cfg.data.dataset_type == "standard":
            dataset_class_to_use = ESC50
            print(f"Fold {test_fold}: Using standard ESC50 dataset for testing.")
        else:
            raise ValueError(f"Unsupported dataset_type: {cfg.data.dataset_type}")

        get_test_dataset = partial(dataset_class_to_use,
                                   subset="test",
                                   root=data_path,
                                   sr=cfg.data.sr,
                                   n_mels=cfg.data.n_mels,
                                   hop_length=cfg.data.hop_length,
                                   n_mfcc=cfg.data.get('n_mfcc', None),
                                   download=cfg.data.get('download', True),
                                   test_folds={test_fold},
                                   global_mean_std=global_stats[test_fold - 1], # Ensure global_stats is available
                                   # Params for InMemoryESC50 / standard ESC50 compatibility
                                   num_aug=0, # No augmentation for test
                                   prob_aug_wave=0.0,
                                   prob_aug_spec=0.0,
                                   val_size=0, # Not used for test set
                                   use_preprocessed_data=cfg.data.get('use_preprocessed_data', False) if dataset_class_to_use == ESC50 else False,
                                   preprocessed_data_root=hyu.to_absolute_path(cfg.data.preprocessed_data_path) if dataset_class_to_use == ESC50 and cfg.data.get('use_preprocessed_data', False) and cfg.data.get('preprocessed_data_path') else None
                                   )

        test_loader = torch.utils.data.DataLoader(get_test_dataset(), # Call partial to get dataset instance
                                                  batch_size=cfg.testing.batch_size, # Use Hydra config
                                                  shuffle=False,
                                                  num_workers=cfg.testing.get('num_workers', 0), # Use Hydra config
                                                  drop_last=False,
                                                  )
        
        # DEBUG: check if testdata is deterministic (multiple testset read, time consuming)
        if check_data_reproducibility: # This var should also come from cfg if needed
            # This check might need re-evaluation as creating two DataLoaders like this is tricky
            # For simplicity, this specific check might be removed or refactored if it causes issues.
            # temp_loader_for_check = torch.utils.data.DataLoader(get_test_dataset(), batch_size=cfg.testing.batch_size, shuffle=False)
            # is_det_file = all([(a[0] == b[0]) for a, b in zip(test_loader, temp_loader_for_check)])
            # is_det_data = all([(a[1] == b[1]).all() for a, b in zip(test_loader, temp_loader_for_check)])
            # is_det_label = all([(a[2] == b[2]).all() for a, b in zip(test_loader, temp_loader_for_check)])
            # assert is_det_file and is_det_data and is_det_label, "test batches not reproducible"
            print("Skipping data reproducibility check in this refactoring step.")


        # tests
        print()
        scores[test_fold] = {}
        for model_file_name in cfg.testing.checkpoints: # Use Hydra config
            model_file = os.path.join(experiment_fold_path, model_file_name) # Use experiment_fold_path
            if not os.path.exists(model_file):
                print(f"Warning: Checkpoint file {model_file} not found. Skipping.")
                continue
            sd = torch.load(model_file, map_location=device)
            model.load_state_dict(sd)
            print('test', model_file)
            # Pass cfg to test function
            test_acc, test_loss, p = test(cfg, model, test_loader,
                                          criterion=criterion, device=device)
            probs[model_file_name].update(p)
            scores[test_fold][model_file_name] = pd.Series(dict(TestAcc=test_acc, TestLoss=np.mean(test_loss)))
            print(scores[test_fold][model_file_name])
        
        if scores[test_fold]: # Check if dict is not empty
            scores[test_fold] = pd.concat(scores[test_fold])
            # Save fold-specific scores in the current Hydra run's output directory (or a subfolder)
            # For now, let's assume test_crossval is run once, and its CWD is the main output dir for this test session.
            # If test_crossval itself is run per fold by an outer script, this path needs care.
            # Assuming test_crossval is run for a specific training output (experiment_root).
            # We can save these detailed fold scores within experiment_root/fold_X or in CWD of test_crossval.
            # Let's save them in the original experiment_fold_path for now.
            scores[test_fold].to_csv(os.path.join(experiment_fold_path, 'test_scores_per_checkpoint.csv'),
                                     index_label=['checkpoint', 'metric'], header=['value'])
        else:
            print(f"No checkpoints found or tested for fold {test_fold}.")

    # Filter out empty fold results before concatenating
    valid_scores = {fold: data for fold, data in scores.items() if not data.empty}
    if not valid_scores:
        print("No valid scores to aggregate.")
        return

    final_scores_df = pd.concat(valid_scores).unstack([-2, -1]) # Adjusted for potentially missing folds
    final_scores_df = pd.concat((final_scores_df, final_scores_df.agg(['mean', 'std'])))
    
    # Save aggregated results in the CWD of this test_crossval run
    # (which is managed by Hydra if test_crossval.py is the entry point)
    output_dir = os.getcwd() # Hydra's current working directory for this run
    
    for model_file_name in cfg.testing.checkpoints:
        file_name_prefix = os.path.splitext(model_file_name)[0]
        # Save probs aggregated across folds for this checkpoint
        # Check if probs[model_file_name] is not empty
        if probs[model_file_name]:
            probs_df = pd.DataFrame(probs[model_file_name]).T
            probs_df.to_csv(os.path.join(output_dir, f'test_probs_{file_name_prefix}_agg.csv'))
        
        # Save scores aggregated across folds for this checkpoint
        if model_file_name in final_scores_df.columns.get_level_values(0): # Check if checkpoint exists in scores
            scores_for_checkpoint_df = final_scores_df[model_file_name]
            scores_for_checkpoint_df.to_csv(os.path.join(output_dir, f'test_scores_{file_name_prefix}_agg.csv'))

    print("\nAggregated Test Results:")
    print(final_scores_df)
    final_scores_df.to_csv(os.path.join(output_dir, 'test_scores_summary_all_checkpoints.csv'))
    print(f"\nAll test outputs saved in: {output_dir}")


if __name__ == "__main__":
    main()
