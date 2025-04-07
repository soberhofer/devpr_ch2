import torch
import torch.nn as nn
import subprocess
import pandas as pd
import numpy as np
import os
import argparse

from dataset.dataset_ESC50 import ESC50, download_extract_zip
from train_crossval import test, make_model, global_stats
import config


if __name__ == "__main__":
    # optional: the test cross validation path can be specified from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('cvpath', nargs='?', default=config.test_experiment)
    args = parser.parse_args()

    reproducible = False
    data_path = config.esc50_path
    use_cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{config.device_id}" if use_cuda else "cpu")

    check_data_reproducibility = False
    if reproducible:
        # improve reproducibility
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(0)
        # for debugging only, uncomment
        #check_data_reproducibility = True

    # digits for logging
    float_fmt = ".3f"
    pd.options.display.float_format = ('{:,' + float_fmt + '}').format
    experiment_root = args.cvpath
    if not os.path.isdir(experiment_root):
        print('download model params')
        download_extract_zip(
            #url='https://cloud.technikum-wien.at/s/9HTN27EADZXGJ72/download/sample-run.zip',
            url='https://cloud.technikum-wien.at/s/PiHsFtnB69cqxPE/download/sample-run.zip',
            file_path=experiment_root + '.zip',
        )


    # instantiate model
    print('*****')
    print("WARNING: Using hardcoded global mean and std. Depends on feature settings!")
    model = make_model()
    model = model.to(device)
    print('*****')

    criterion = nn.CrossEntropyLoss().to(device)

    # for all folds
    scores = {}
    probs = {model_file_name: {} for model_file_name in config.test_checkpoints}
    for test_fold in config.test_folds:
        experiment = os.path.join(experiment_root, f'{test_fold}')

        test_loader = torch.utils.data.DataLoader(ESC50(subset="test", test_folds={test_fold},
                                                        global_mean_std=global_stats[test_fold - 1],
                                                        root=data_path, download=True),
                                                  batch_size=config.batch_size,
                                                  shuffle=False,
                                                  num_workers=0,  # config.num_workers,
                                                  drop_last=False,
                                                  )
        # DEBUG: check if testdata is deterministic (multiple testset read, time consuming)
        if check_data_reproducibility:
            is_det_file = all([(a[0] == b[0]) for a, b in zip(test_loader, test_loader)])
            is_det_data = all([(a[1] == b[1]).all() for a, b in zip(test_loader, test_loader)])
            is_det_label = all([(a[2] == b[2]).all() for a, b in zip(test_loader, test_loader)])
            assert is_det_file and is_det_data and is_det_label, "test batches not reproducible"

        # tests
        print()
        scores[test_fold] = {}
        for model_file_name in config.test_checkpoints:
            model_file = os.path.join(experiment, model_file_name)
            sd = torch.load(model_file, map_location=device)
            model.load_state_dict(sd)
            print('test', model_file)
            test_acc, test_loss, p = test(model, test_loader,
                                          criterion=criterion, device=device)
            probs[model_file_name].update(p)
            scores[test_fold][model_file_name] = pd.Series(dict(TestAcc=test_acc, TestLoss=np.mean(test_loss)))
            print(scores[test_fold][model_file_name])
        scores[test_fold] = pd.concat(scores[test_fold])
        scores[test_fold].to_csv(os.path.join(experiment, 'test_scores.csv'),
                                 index_label=['checkpoint', 'metric'], header=['value'])
        # print(scores[test_fold].unstack())
    scores = pd.concat(scores).unstack([-2, -1])
    scores = pd.concat((scores, scores.agg(['mean', 'std'])))
    for model_file_name in config.test_checkpoints:
        file_name = os.path.splitext(model_file_name)[0]
        file_path = os.path.join(experiment_root, f'test_probs_{file_name}.csv')
        probs[model_file_name] = pd.DataFrame(probs[model_file_name]).T
        probs[model_file_name].to_csv(file_path)
        file_path = os.path.join(experiment_root, f'test_scores_{file_name}.csv')
        scores[model_file_name].to_csv(file_path)
    print(scores)
    print()
