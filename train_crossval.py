import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import dataloader
import pandas as pd
import numpy as np
import os
import datetime
from tqdm import tqdm
import sys
# import argparse # Removed argparse
from functools import partial
import wandb
import json
import hydra
from omegaconf import DictConfig, OmegaConf
import hydra.utils as hyu # Use hyu for hydra utils



from models.model_classifier import AudioMLP, AudioCNN, TFCNN, TFCNN2
from models.tfcnn import TFNet, Cnn
from models.mobilenet import mobilenet_v3_large, mobilenet_v3_small # Keep V3 imports
from models.mobilenetv2 import MobileNetV2Audio # Import V2 from its new file
from models.resnet import ResNet50, ResNet18 # Import ResNet50 and ResNet18
from models.romnet import Romnet # Import Romnet
from models.utils import EarlyStopping, Tee
from dataset.dataset_ESC50 import ESC50, InMemoryESC50, calculate_fold_descriptive_stats # Import InMemoryESC50 and new stats function
# import config # Removed old config import


# evaluate model on different testing data 'dataloader'
# Added cfg parameter
def test(cfg: DictConfig, model, dataloader, criterion, device):
    model.eval()

    losses = []
    corrects = 0
    samples_count = 0
    probs = {}
    with torch.no_grad():
        # no gradient computation needed
        for k, x, label in tqdm(dataloader, unit='bat', disable=cfg.training.disable_bat_pbar, position=0): # Use cfg
            x = x.float().to(device)
            y_true = label.to(device)

            
            #the forward pass through the model
            y_prob = model(x)

            loss = criterion(y_prob, y_true)
            losses.append(loss.item())

            y_pred = torch.argmax(y_prob, dim=1)
            corrects += (y_pred == y_true).sum().item()
            samples_count += y_true.shape[0]
            for w, p in zip(k, y_prob):
                probs[w] = [float(v) for v in p]

    acc = corrects / samples_count
    return acc, losses, probs


# Added cfg, model, train_loader, criterion, optimizer parameters
def train_epoch(cfg: DictConfig, model, train_loader, criterion, optimizer, device):
    # switch to training
    model.train()

    losses = []
    corrects = 0
    samples_count = 0
    for _, x, label in tqdm(train_loader, unit='bat', disable=cfg.training.disable_bat_pbar, position=0): # Use cfg
        x = x.float().to(device)
        y_true = label.to(device)

        # the forward pass through the model
        y_prob = model(x)

        # we could also use 'F.one_hot(y_true)' for 'y_true', but this would be slower
        loss = criterion(y_prob, y_true)
        # reset the gradients to zero - avoids accumulation
        optimizer.zero_grad()
        # compute the gradient with backpropagation
        loss.backward()
        losses.append(loss.item())
        # minimize the loss via the gradient - adapts the model parameters
        optimizer.step()

        y_pred = torch.argmax(y_prob, dim=1)
        corrects += (y_pred == y_true).sum().item()
        samples_count += y_true.shape[0]

    acc = corrects / samples_count
    return acc, losses


# Added cfg, model, train_loader, val_loader, criterion, optimizer, scheduler, device, fold_output_dir, float_fmt parameters
# Note: 'run' object is not passed here, we check cfg directly
def fit_classifier(cfg: DictConfig, model, train_loader, val_loader, criterion, optimizer, scheduler, device, fold_output_dir, float_fmt):
    num_epochs = cfg.training.epochs # Use cfg

    # Checkpoints are saved in the fold-specific output directory
    best_val_loss_path = os.path.join(fold_output_dir, 'best_val_loss.pt')
    terminal_path = os.path.join(fold_output_dir, 'terminal.pt')

    loss_stopping = EarlyStopping(patience=cfg.training.patience, delta=0.002, verbose=True, float_fmt=float_fmt, # Use cfg
                                  checkpoint_file=best_val_loss_path)

    pbar = tqdm(range(1, 1 + num_epochs), ncols=50, unit='ep', file=sys.stdout, ascii=True)
    for epoch in (range(1, 1 + num_epochs)):
        # iterate once over training data
        # Pass necessary arguments to train_epoch
        train_acc, train_loss = train_epoch(cfg, model, train_loader, criterion, optimizer, device)

        # validate model
        # Pass cfg to test function
        val_acc, val_loss, _ = test(cfg, model, val_loader, criterion=criterion, device=device)
        val_loss_avg = np.mean(val_loss)

        # print('\n')
        pbar.update()
        # pbar.refresh() syncs output when pbar on stderr
        # pbar.refresh()
        lr = scheduler.get_last_lr()
        print(end=' ')
        print(  # f" Epoch: {epoch}/{num_epochs}",
            f"LR:{lr[0]:.3e}",
            f"TrnAcc={train_acc:{float_fmt}}",
            f"ValAcc={val_acc:{float_fmt}}",
            f"TrnLoss={np.mean(train_loss):{float_fmt}}",
            f"ValLoss={val_loss_avg:{float_fmt}}",
            end=' ')

        # Log to wandb if enabled
        if cfg.use_wandb:
            wandb.log({
                "LearningRate": lr[0],
                "ValLoss": val_loss_avg,
                "ValAcc": val_acc,
                "TrnLoss": np.mean(train_loss),
                "TrnAcc": train_acc,
            })

        early_stop, improved = loss_stopping(val_loss_avg, model, epoch)
        if not improved:
            print()
        if early_stop:
            print("Early stopping")
            break

        # advance the optimization scheduler
        scheduler.step()
    # save full model
    torch.save(model.state_dict(), terminal_path) # Use updated path
    # wandb.save(terminal_path) # Wandb automatically saves files in its run dir, this might be redundant

    # run.finish() # Wandb run finishing is handled outside this function now


# build model from configuration using cfg
def make_model(cfg: DictConfig):
    model_type = cfg.model.name
    params = cfg.model.params
    # n_mels and output_size are interpolated in the config file itself
    # No need to pass them separately if using ${data.n_mels} and ${data.n_classes}

    if model_type == 'AudioMLP':
        # Pass parameters directly from cfg.model.params
        model = AudioMLP(n_steps=params.n_steps, n_mels=params.n_mels,
                         hidden1_size=params.hidden1_size, hidden2_size=params.hidden2_size,
                         output_size=params.output_size, time_reduce=params.time_reduce)
    elif model_type == 'AudioCNN':
         # Assuming AudioCNN takes n_mels and output_size, adjust if needed
        model = AudioCNN(n_mels=params.n_mels, output_size=params.output_size)
    elif model_type == 'tfcnn':
        # Assuming TFCNN takes num_classes, adjust if needed
        model = TFCNN(num_classes=params.output_size)
    elif model_type == 'hpss':
         # Assuming TFCNN2 takes n_mels and output_size, adjust if needed
        model = TFCNN2(n_mels=params.n_mels, output_size=params.output_size)
    elif model_type == 'tfnet':
        # Assuming TFNet takes classes_num, adjust if needed
        model = TFNet(classes_num=params.output_size) #, in_channels=1)
    elif model_type == 'tfnet_cnn':
        # Assuming Cnn takes classes_num, adjust if needed
        model = Cnn(classes_num=params.output_size)
    elif model_type == 'mobilenetv2':
        # Instantiate MobileNetV2Audio using parameters from config
        model = MobileNetV2Audio(num_classes=params.output_size,
                                 pretrained=params.get('pretrained', False), # Use V2 config params
                                 input_channels=params.get('input_channels', 1),
                                 dropout_prob=params.get('dropout_prob', 0.5)) # Use V2 config params
    elif model_type == 'mobilenet_v3_large':
        # Instantiate MobileNetV3 Large using parameters from config
        model = mobilenet_v3_large(num_classes=params.output_size,
                                   input_channels=params.get('input_channels', 1), # Default to 1 input channel
                                   dropout=params.get('dropout_prob', 0.2)) # Use dropout from config, default 0.2
    elif model_type == 'mobilenet_v3_small':
        # Instantiate MobileNetV3 Small using parameters from config
        model = mobilenet_v3_small(num_classes=params.output_size,
                                   input_channels=params.get('input_channels', 1), # Default to 1 input channel
                                   dropout=params.get('dropout_prob', 0.2)) # Use dropout from config, default 0.2
    elif model_type == 'ResNet50':
        # Instantiate ResNet50 using parameters from config
        # params.output_size (interpolated from data.n_classes) and params.channels are defined in conf/model/resnet50.yaml
        model = ResNet50(num_classes=params.output_size, # Changed from params.num_classes
                         channels=params.get('channels', 1), # Added .get() with default value 1
                         dropout_prob=params.get('dropout_prob', 0.0))
    elif model_type == 'ResNet18':
        # Instantiate ResNet18 using parameters from config
        model = ResNet18(num_classes=params.output_size,
                         channels=params.get('channels', 1),
                         dropout_prob=params.get('dropout_prob', 0.0))
    elif model_type == 'Romnet':
        # Instantiate Romnet using parameters from config
        model = Romnet(num_classes=params.output_size,
                       channels=params.get('channels', 1), # Default to 1 input channel
                       dropout_prob=params.get('dropout_prob', 0.2)) # Default dropout_prob to 0.2
    else:
        raise ValueError(f"Invalid model type in config: {model_type}")
    return model


# Main training logic wrapped in a function decorated by Hydra
@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Print the configuration - useful for debugging
    print(OmegaConf.to_yaml(cfg))

    # --- Setup ---
    # Use absolute path for data relative to the original working directory
    # Hydra changes the working directory, so we need this.
    data_path = hyu.to_absolute_path(cfg.data.path)
    n_classes = cfg.data.n_classes # Use cfg

    # Determine device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{cfg.training.device_id}") # Use cfg
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f'Using device: {device}')

    # Formatting for logs
    float_fmt = ".3f"
    pd.options.display.float_format = ('{:,' + float_fmt + '}').format

    # Hydra automatically creates and manages the output directory (current working directory)
    # No need for experiment_root or manually creating fold directories within the script
    # The output directory for the current fold will be the CWD set by Hydra.
    # We can get the original CWD if needed: original_cwd = hyu.get_original_cwd()

    # --- Cross-validation Loop ---
    scores = {}
    # Use cfg.data.folds (which is 5 in esc50.yaml) instead of cfg.data.num_folds
    all_available_folds = set(range(1, cfg.data.folds + 1)) 

    # Iterate through test folds defined in the config
    for test_fold in cfg.data.test_folds:
        print(f"\n===== FOLD {test_fold} =====")

        # Hydra manages output directories per run/job.
        # For cross-validation, we might want outputs grouped by fold.
        # One way is to let Hydra handle the main run directory, and we save fold-specific things inside.
        # Or, structure the Hydra launch itself (e.g., using hydra-optuna-sweeper or custom launcher).
        # For simplicity here, we'll use the Hydra CWD for each fold's run.
        # If running folds sequentially, Hydra creates a new dir each time.
        # If running in parallel (e.g., via sweep), Hydra handles subdirs.
        current_run_dir = os.getcwd() # Hydra sets this (main output dir for the entire run if not sweeping over folds)
        
        # Create a specific output directory for this fold's artifacts
        fold_output_dir = os.path.join(current_run_dir, str(test_fold))
        os.makedirs(fold_output_dir, exist_ok=True)
        print(f"Output directory for fold {test_fold}: {fold_output_dir}")

        # Initialize WandB for the current fold if enabled
        run = None # Initialize run to None
        if cfg.use_wandb:
            now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") # Define now_str
            run_name = f"{cfg.model.name}-fold{test_fold}-{cfg.get('comment', '')}-{now_str}" # Use now_str
            run = wandb.init(
                project=cfg.get("wandb_project", "challenge2"), # Make project configurable
                name=run_name,
                config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True) # Log Hydra config
            )

        # Clone stdout/stderr to a log file in the Hydra output directory
        log_file_path = os.path.join(current_run_dir, f'train_fold_{test_fold}.log')
        with Tee(log_file_path, 'w', 1, encoding='utf-8', newline='\n', proc_cr=True):

            # --- Data Loading ---
            # Use absolute data path and cfg for parameters
            
            dataset_class_to_use = None
            if cfg.data.dataset_type == "in_memory":
                dataset_class_to_use = InMemoryESC50
                print("Using InMemoryESC50 dataset.")
            elif cfg.data.dataset_type == "standard":
                dataset_class_to_use = ESC50
                print("Using standard ESC50 dataset.")
            else:
                raise ValueError(f"Unsupported dataset_type: {cfg.data.dataset_type}")

            # Pass required parameters from cfg to the selected dataset constructor via partial
            get_fold_dataset = partial(dataset_class_to_use,
                                       root=data_path, # Original data path, used for download if needed
                                       sr=cfg.data.sr,
                                       n_mels=cfg.data.n_mels,
                                       hop_length=cfg.data.hop_length,
                                       val_size=cfg.training.val_size,
                                       n_mfcc=cfg.data.get('n_mfcc', None), # Pass n_mfcc if defined
                                       download=cfg.data.get('download', True), # Make download configurable, default True
                                       test_folds={test_fold},
                                       prob_aug_wave=cfg.data.get('prob_aug_wave', 0.0),
                                       prob_aug_spec=cfg.data.get('prob_aug_spec', 0.0),
                                       # global_mean_std will be set based on dataset_class_to_use
                                       num_aug=cfg.data.num_aug, # Pass num_aug
                                       # Conditional parameters for standard ESC50's external preprocessing
                                       use_preprocessed_data=cfg.data.get('use_preprocessed_data', False) if dataset_class_to_use == ESC50 else False,
                                       preprocessed_data_root=hyu.to_absolute_path(cfg.data.preprocessed_data_path) if dataset_class_to_use == ESC50 and cfg.data.get('use_preprocessed_data', False) and cfg.data.get('preprocessed_data_path') else None
                                       )
            
            # Determine current training folds for stats calculation
            current_train_folds_for_stats = all_available_folds - {test_fold}
            
            # Calculate or load stats for the current training folds
            # Note: Add caching mechanism here if desired for performance
            fold_mean, fold_std = calculate_fold_descriptive_stats(cfg, data_path, current_train_folds_for_stats, all_available_folds)
            current_fold_global_mean_std = (fold_mean, fold_std)

            # Update the partial function with the dynamically calculated stats
            if dataset_class_to_use == InMemoryESC50:
                # InMemoryESC50 expects global_mean_std_for_norm for applying normalization after loading from cache
                get_fold_dataset = partial(get_fold_dataset.func, **get_fold_dataset.keywords, global_mean_std_for_norm=current_fold_global_mean_std)
            else: # For standard ESC50
                get_fold_dataset = partial(get_fold_dataset.func, **get_fold_dataset.keywords, global_mean_std=current_fold_global_mean_std)


            train_set = get_fold_dataset(subset="train")
            print('*****')
            print(f'Train folds: {train_set.train_folds}, Test fold: {train_set.test_folds}')
            # print('random wave cropping') # Assuming this happens inside ESC50 dataset

            train_loader = torch.utils.data.DataLoader(train_set,
                                                       batch_size=cfg.training.batch_size, # Use cfg
                                                       shuffle=True,
                                                       num_workers=cfg.training.num_workers, # Use cfg
                                                       drop_last=False,
                                                       persistent_workers=cfg.training.persistent_workers, # Use cfg
                                                       pin_memory=True,
                                                       )

            val_loader = torch.utils.data.DataLoader(get_fold_dataset(subset="val"),
                                                     batch_size=cfg.training.batch_size, # Use cfg
                                                     shuffle=False,
                                                     num_workers=cfg.training.num_workers, # Use cfg
                                                     drop_last=False,
                                                     persistent_workers=cfg.training.persistent_workers, # Use cfg
                                                     )

            # --- Model, Loss, Optimizer, Scheduler ---
            print()
            model = make_model(cfg).to(device) # Use cfg
            print(f"Model: {cfg.model.name}")
            print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
            print('*****')

            criterion = nn.CrossEntropyLoss(label_smoothing=cfg.training.label_smoothing).to(device)
            print(f"Using label smoothing: {cfg.training.label_smoothing}") # Add log message

            # Optimizer
            if cfg.training.optimizer == 'Adam':
                optimizer = torch.optim.Adam(model.parameters(),
                                             lr=cfg.training.lr, # Use cfg
                                             weight_decay=cfg.training.weight_decay) # Use cfg
            elif cfg.training.optimizer == 'AdamW':
                optimizer = torch.optim.AdamW(model.parameters(),
                                              lr=cfg.training.lr, # Use cfg
                                              weight_decay=cfg.training.weight_decay) # Use cfg
            elif cfg.training.optimizer == 'SGD':
                 optimizer = torch.optim.SGD(model.parameters(),
                                             lr=cfg.training.lr, # Use cfg
                                             momentum=0.9, # Consider adding momentum to config if needed
                                             weight_decay=cfg.training.weight_decay) # Use cfg
            else:
                raise ValueError(f"Unsupported optimizer: {cfg.training.optimizer}")

            # Scheduler (Example: adapting OneCycleLR)
            # Ensure scheduler config matches the type used
            if cfg.training.scheduler.type == 'OneCycleLR':
                # Calculate total steps correctly: epochs * steps_per_epoch
                steps_per_epoch = len(train_loader)
                total_steps = cfg.training.epochs * steps_per_epoch
                print(f"Scheduler: OneCycleLR, Total Steps: {total_steps}") # Add log
                scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=cfg.training.scheduler.max_lr, # Add max_lr to config if using OneCycleLR
                    total_steps=total_steps, # Corrected total_steps
                    pct_start=cfg.training.scheduler.pct_start, # Add pct_start to config
                    # anneal_strategy='cos', # Consider adding anneal_strategy if needed
                )
            elif cfg.training.scheduler.type == 'StepLR':
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                            step_size=cfg.training.scheduler.step_size, # Use cfg
                                                            gamma=cfg.training.scheduler.gamma, # Use cfg
                )
            elif cfg.training.scheduler.type == 'CyclicLR':
                # Ensure required parameters exist in the config
                required_cyclic_params = ['base_lr', 'max_lr', 'step_size_up']
                if not all(hasattr(cfg.training.scheduler, p) for p in required_cyclic_params):
                    raise ValueError(f"Missing required parameters for CyclicLR: {required_cyclic_params}")

                scheduler = torch.optim.lr_scheduler.CyclicLR(
                    optimizer,
                    base_lr=cfg.training.scheduler.base_lr,
                    max_lr=cfg.training.scheduler.max_lr,
                    step_size_up=cfg.training.scheduler.step_size_up,
                    # Optional parameters from config, using .get() for safety
                    mode=cfg.training.scheduler.get('mode', 'triangular'),
                    gamma=cfg.training.scheduler.get('gamma', 1.0),
                    cycle_momentum=cfg.training.scheduler.get('cycle_momentum', True),
                    base_momentum=cfg.training.scheduler.get('base_momentum', 0.8),
                    max_momentum=cfg.training.scheduler.get('max_momentum', 0.9),
                 )
            elif cfg.training.scheduler.type == 'CosineAnnealingLR':
                # Calculate T_max as total training iterations
                steps_per_epoch = len(train_loader)
                T_max = cfg.training.epochs * steps_per_epoch
                print(f"Scheduler: CosineAnnealingLR, T_max: {T_max}, eta_min: {cfg.training.scheduler.eta_min}") # Add log
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=cfg.training.epochs, # Maximum number of iterations.
                    eta_min=cfg.training.scheduler.eta_min, # Minimum learning rate.
                )
            else:
                 # Add other schedulers or a default/None option
                 raise ValueError(f"Unsupported scheduler type: {cfg.training.scheduler.type}")


            # --- Training ---
            print("\n--- Starting Training ---")
            # Pass fold_output_dir to fit_classifier
            fit_classifier(cfg, model, train_loader, val_loader, criterion, optimizer, scheduler, device, fold_output_dir, float_fmt)
            print("--- Training Finished ---")


            # --- Testing ---
            # This internal testing block evaluates the model trained for the current fold.
            # test_crossval.py is for testing pre-existing models from a completed training run.
            print("\n--- Starting Testing for current fold ---")
            test_loader = torch.utils.data.DataLoader(get_fold_dataset(subset="test"),
                                                      batch_size=cfg.training.batch_size, # Use cfg (or define test batch size)
                                                      shuffle=False,
                                                      num_workers=cfg.training.num_workers, # Use cfg num_workers for consistency
                                                      drop_last=False,
                                                      )

            # Load the best model saved by EarlyStopping for this fold for testing
            best_model_fold_path = os.path.join(fold_output_dir, 'best_val_loss.pt') # Path to this fold's best model
            if os.path.exists(best_model_fold_path):
                 print(f"Loading best model for fold {test_fold} from: {best_model_fold_path}")
                 model.load_state_dict(torch.load(best_model_fold_path)) # Load best model state
            else:
                 print("Warning: best_val_loss.pt not found. Testing with the terminal model.")
                 # If best model isn't found, the model variable holds the terminal state

            # Pass cfg to test function
            test_acc, test_loss, _ = test(cfg, model, test_loader, criterion=criterion, device=device)
            scores[test_fold] = pd.Series(dict(TestAcc=test_acc, TestLoss=np.mean(test_loss)))
            print(f"Fold {test_fold} Test Results:")
            print(scores[test_fold])
            print("--- Testing Finished ---")

            # Log final fold scores to wandb if enabled
            if run: # Check if run was initialized
                wandb.log({f"final_fold_{test_fold}_test_acc": test_acc,
                           f"final_fold_{test_fold}_test_loss": np.mean(test_loss)})

        # Finish WandB run for the current fold if enabled
        if run: # Check if run was initialized
            run.finish()

    # --- Aggregate Results ---
    print("\n===== CROSS-VALIDATION RESULTS =====")
    scores_df = pd.concat(scores, axis=1).T # Transpose to have folds as rows
    # scores_df = pd.concat(scores).unstack([-1]) # Original way
    agg_scores = scores_df.agg(['mean', 'std'])
    final_scores_df = pd.concat([scores_df, agg_scores])
    print(final_scores_df)

    # Save aggregated scores to a file in the original working directory or a specific results dir
    # Note: Hydra's multirun directory might be a better place if running a sweep.
    # For a simple sequential run, saving outside the fold-specific Hydra dirs might be desired.
    try:
        # Save in the directory where the script was launched
        original_cwd = hyu.get_original_cwd()
        scores_path = os.path.join(original_cwd, 'crossval_scores.csv')
        final_scores_df.to_csv(scores_path)
        print(f"Saved cross-validation scores to: {scores_path}")
        # Log aggregated scores to a final summary (optional, needs separate wandb run or different handling)
        # wandb.log({"mean_test_acc": agg_scores.loc['mean', 'TestAcc'], ...})
    except Exception as e:
        print(f"Error saving final scores: {e}")


if __name__ == "__main__":
    # The @hydra.main decorator handles execution.
    # Any code here will run *before* Hydra takes over if not careful.
    # It's generally best to put all logic inside the decorated function.
    main()
