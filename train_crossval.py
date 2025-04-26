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
from models.utils import EarlyStopping, Tee
from dataset.dataset_ESC50 import ESC50
# import config # Removed old config import



# mean and std of train data for every fold
global_stats = np.array([[-54.364834, 20.853344],
                         [-54.279022, 20.847532],
                         [-54.18343, 20.80387],
                         [-54.223698, 20.798292],
                         [-54.200905, 20.949806]])

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


# Added cfg, model, train_loader, val_loader, criterion, optimizer, scheduler, device, experiment_dir, float_fmt parameters
def fit_classifier(cfg: DictConfig, model, train_loader, val_loader, criterion, optimizer, scheduler, device, experiment_dir, float_fmt):
    num_epochs = cfg.training.epochs # Use cfg

    # Use Hydra's current working directory for checkpoints
    best_val_loss_path = os.path.join(experiment_dir, 'best_val_loss.pt')
    terminal_path = os.path.join(experiment_dir, 'terminal.pt')

    loss_stopping = EarlyStopping(patience=cfg.training.patience, delta=0.002, verbose=True, float_fmt=float_fmt, # Use cfg
                                  checkpoint_file=best_val_loss_path) # Use updated path

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
    print("WARNING: Using hardcoded global mean and std. Depends on feature settings!") # Keep warning for now

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
        current_run_dir = os.getcwd() # Hydra sets this
        print(f"Output directory for fold {test_fold}: {current_run_dir}")

        # Initialize WandB for the current fold
        # Use Hydra config for naming and log the config
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
            # Pass required parameters from cfg to ESC50 constructor via partial
            get_fold_dataset = partial(ESC50,
                                       root=data_path,
                                       sr=cfg.data.sr,
                                       n_mels=cfg.data.n_mels,
                                       hop_length=cfg.data.hop_length,
                                       val_size=cfg.training.val_size,
                                       # n_mfcc=cfg.data.get('n_mfcc', None), # Pass n_mfcc if defined in data config
                                       download=True, # Keep download=True? Or make configurable?
                                       test_folds={test_fold},
                                       global_mean_std=global_stats[test_fold - 1]) # Still using hardcoded stats

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

            criterion = nn.CrossEntropyLoss().to(device)

            # Optimizer
            if cfg.training.optimizer == 'Adam':
                optimizer = torch.optim.Adam(model.parameters(),
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
                 scheduler = torch.optim.lr_scheduler.OneCycleLR(
                     optimizer,
                     max_lr=cfg.training.scheduler.max_lr, # Add max_lr to config if using OneCycleLR
                     total_steps=cfg.training.epochs, # Simplified total_steps, adjust if needed
                     pct_start=cfg.training.scheduler.pct_start, # Add pct_start to config
                 )
            elif cfg.training.scheduler.type == 'StepLR':
                 scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                             step_size=cfg.training.scheduler.step_size, # Use cfg
                                                             gamma=cfg.training.scheduler.gamma) # Use cfg
            else:
                 # Add other schedulers or a default/None option
                 raise ValueError(f"Unsupported scheduler type: {cfg.training.scheduler.type}")


            # --- Training ---
            print("\n--- Starting Training ---")
            # Pass all required arguments to fit_classifier
            fit_classifier(cfg, model, train_loader, val_loader, criterion, optimizer, scheduler, device, current_run_dir, float_fmt)
            print("--- Training Finished ---")


            # --- Testing ---
            print("\n--- Starting Testing ---")
            test_loader = torch.utils.data.DataLoader(get_fold_dataset(subset="test"),
                                                      batch_size=cfg.training.batch_size, # Use cfg (or define test batch size)
                                                      shuffle=False,
                                                      num_workers=cfg.training.num_workers, # Use cfg num_workers for consistency
                                                      drop_last=False,
                                                      )

            # Load the best model saved by EarlyStopping for testing
            best_model_path = os.path.join(current_run_dir, 'best_val_loss.pt')
            if os.path.exists(best_model_path):
                 print(f"Loading best model from: {best_model_path}")
                 model.load_state_dict(torch.load(best_model_path)) # Load best model state
            else:
                 print("Warning: best_val_loss.pt not found. Testing with the terminal model.")
                 # If best model isn't found, the model variable holds the terminal state

            # Pass cfg to test function
            test_acc, test_loss, _ = test(cfg, model, test_loader, criterion=criterion, device=device)
            scores[test_fold] = pd.Series(dict(TestAcc=test_acc, TestLoss=np.mean(test_loss)))
            print(f"Fold {test_fold} Test Results:")
            print(scores[test_fold])
            print("--- Testing Finished ---")

            # Log final fold scores to wandb
            wandb.log({f"final_fold_{test_fold}_test_acc": test_acc,
                       f"final_fold_{test_fold}_test_loss": np.mean(test_loss)})

        # Finish WandB run for the current fold
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
