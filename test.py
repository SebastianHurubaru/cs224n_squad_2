"""Test a model and generate submission CSV.

Usage:
    > python test.py --split SPLIT --load_path PATH --name NAME
    where
    > SPLIT is either "dev" or "test"
    > PATH is a path to a checkpoint (e.g., save/train/model-01/best.pth.tar)
    > NAME is a name to identify the test run

Author:
    Chris Chute (chute@stanford.edu)
"""

import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import util
import glob

from args import get_test_args
from collections import OrderedDict
from json import dumps
from models import BiDAF, BiDAFExtra, FusionNet
from os.path import join
from tensorboardX import SummaryWriter
from tqdm import tqdm
from ujson import load as json_load
from util import collate_fn, SQuAD


def main(args):
    # Set up logging
    args.save_dir = util.get_save_dir(args.save_dir, args.name, training=False)
    log = util.get_logger(args.save_dir, args.name)
    device, gpu_ids = util.get_available_devices()
    args.batch_size *= max(1, len(gpu_ids))
    log.info(f'Args: {dumps(vars(args), indent=4, sort_keys=True)}')

    # Get embeddings
    log.info('Loading embeddings...')
    word_vectors = util.torch_from_json(args.word_emb_file)


    models = {}

    if args.use_ensemble:

        total_models = 0
        for model_name in ['bidaf', 'bidafextra', 'fusionnet']:

            models_list = []

            for model_file in glob.glob(f'{args.load_path}/{model_name}-*/{args.ensemble_models}'):

                # Get model
                log.info('Building model...')
                if model_name == 'bidaf':
                    model = BiDAF(word_vectors=word_vectors,
                                  hidden_size=args.hidden_size)
                elif model_name == 'bidafextra':
                    model = BiDAFExtra(word_vectors=word_vectors,
                                       args=args)
                elif model_name == 'fusionnet':
                    model = FusionNet(word_vectors=word_vectors,
                                      args=args)

                model = nn.DataParallel(model, gpu_ids)
                log.info(f'Loading checkpoint from {model_file}...')
                model = util.load_model(model, model_file, gpu_ids, return_step=False)

                # Load each model on CPU (have plenty of RAM ...)
                model = model.cpu()
                model.eval()

                models_list.append(model)

            models[model_name] = models_list

            total_models += len(models_list)

        log.info(f'Using an ensemble of {total_models} models')

    else:

        device, gpu_ids = util.get_available_devices()

        # Get model
        log.info('Building model...')
        if args.model == 'bidaf':
            model = BiDAF(word_vectors=word_vectors,
                          hidden_size=args.hidden_size)
        elif args.model == 'bidafextra':
            model = BiDAFExtra(word_vectors=word_vectors,
                               args=args)
        elif args.model == 'fusionnet':
            model = FusionNet(word_vectors=word_vectors,
                              args=args)

        model = nn.DataParallel(model, gpu_ids)
        log.info(f'Loading checkpoint from {args.load_path}...')
        model = util.load_model(model, args.load_path, gpu_ids, return_step=False)
        model = model.to(device)
        model.eval()

        models[args.model] = [model]


    # Get data loader
    log.info('Building dataset...')
    record_file = vars(args)[f'{args.split}_record_file']
    dataset = SQuAD(record_file, args)
    data_loader = data.DataLoader(dataset,
                                  batch_size=args.batch_size,
                                  shuffle=False,
                                  num_workers=args.num_workers,
                                  collate_fn=collate_fn)

    # Evaluate
    log.info(f'Evaluating on {args.split} split...')
    nll_meter = util.AverageMeter()
    pred_dict = {}  # Predictions for TensorBoard
    sub_dict = {}   # Predictions for submission
    eval_file = vars(args)[f'{args.split}_eval_file']
    with open(eval_file, 'r') as fh:
        gold_dict = json_load(fh)
    with torch.no_grad(), \
            tqdm(total=len(dataset)) as progress_bar:
        for cw_idxs, cc_idxs, qw_idxs, qc_idxs, cw_pos, cw_ner, cw_freq, cqw_extra, y1, y2, ids in data_loader:
            # Setup for forward
            cw_idxs = cw_idxs.to(device)
            qw_idxs = qw_idxs.to(device)
            batch_size = cw_idxs.size(0)

            p1s = []
            p2s = []

            for model_name in models:
                for model in models[model_name]:
                    # Move model to GPU to evaluate
                    model = model.to(device)

                    # Forward
                    if model_name == 'bidaf':
                        log_p1, log_p2 = model.to(device)(cw_idxs, qw_idxs)
                    else:
                        log_p1, log_p2 = model.to(device)(cw_idxs, qw_idxs, cw_pos, cw_ner, cw_freq, cqw_extra)

                    log_p1, log_p2 = log_p1.cpu(), log_p2.cpu()

                    if not args.use_ensemble:
                        y1, y2 = y1.to(device), y2.to(device)
                        loss = F.nll_loss(log_p1, y1) + F.nll_loss(log_p2, y2)
                        nll_meter.update(loss.item(), batch_size)

                    # Move model back to CPU to release GPU memory
                    model = model.cpu()

                    # Get F1 and EM scores
                    p1, p2 = log_p1.exp().unsqueeze(-1).cpu(), log_p2.exp().unsqueeze(-1).cpu()
                    p1s.append(p1), p2s.append(p2)

            best_ps = torch.max(torch.cat([torch.cat(p1s, -1).unsqueeze(-1), torch.cat(p2s, -1).unsqueeze(-1)], -1), -2)[0]

            p1, p2 = best_ps[:, :, 0], best_ps[:, :, 1]
            starts, ends = util.discretize(p1, p2, args.max_ans_len, args.use_squad_v2)

            # Log info
            progress_bar.update(batch_size)
            if args.split != 'test':
                # No labels for the test set, so NLL would be invalid
                progress_bar.set_postfix(NLL=nll_meter.avg)

            idx2pred, uuid2pred = util.convert_tokens(gold_dict,
                                                      ids.tolist(),
                                                      starts.tolist(),
                                                      ends.tolist(),
                                                      args.use_squad_v2)
            pred_dict.update(idx2pred)
            sub_dict.update(uuid2pred)

    # Log results (except for test set, since it does not come with labels)
    if args.split != 'test':
        results = util.eval_dicts(gold_dict, pred_dict, args.use_squad_v2)
        results_list = [('NLL', nll_meter.avg),
                        ('F1', results['F1']),
                        ('EM', results['EM'])]
        if args.use_squad_v2:
            results_list.append(('AvNA', results['AvNA']))
        results = OrderedDict(results_list)

        # Log to console
        results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in results.items())
        log.info(f'{args.split.title()} {results_str}')

        # Log to TensorBoard
        tbx = SummaryWriter(args.save_dir)
        util.visualize(tbx,
                       pred_dict=pred_dict,
                       eval_path=eval_file,
                       step=0,
                       split=args.split,
                       num_visuals=args.num_visuals)

    # Write submission file
    sub_path = join(args.save_dir, args.split + '_' + args.sub_file)
    log.info(f'Writing submission file to {sub_path}...')
    with open(sub_path, 'w', newline='', encoding='utf-8') as csv_fh:
        csv_writer = csv.writer(csv_fh, delimiter=',')
        csv_writer.writerow(['Id', 'Predicted'])
        for uuid in sorted(sub_dict):
            csv_writer.writerow([uuid, sub_dict[uuid]])


if __name__ == '__main__':
    main(get_test_args())
