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
from collections import defaultdict

from args import get_test_args
from collections import OrderedDict
from json import dumps
from models import BiDAF, BiDAFExtra, FusionNet
from os.path import join
from tensorboardX import SummaryWriter
from tqdm import tqdm
from ujson import load as json_load
from util import collate_fn, SQuAD
from collections import OrderedDict


def find_first_interrogative_pronoun(list_of_interrogative_pronouns, list_of_words_question):
    for word_question in list_of_words_question:
        if word_question in list_of_interrogative_pronouns:
            return word_question
    return ""


def main(args):
    # Set up logging
    args.save_dir = util.get_save_dir(args.save_dir, args.name, training=False)
    log = util.get_logger(args.save_dir, args.name)
    log.info(f'Args: {dumps(vars(args), indent=4, sort_keys=True)}')
    device, gpu_ids = util.get_available_devices()
    args.batch_size *= max(1, len(gpu_ids))

    # Get embeddings
    log.info('Loading embeddings...')
    word_vectors = util.torch_from_json(args.word_emb_file)

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

    # Get data loader
    log.info('Building dataset...')
    record_file = vars(args)[f'{args.split}_record_file']
    dataset = SQuAD(record_file, args)
    data_loader = data.DataLoader(dataset,
                                  batch_size=args.batch_size,
                                  shuffle=False,
                                  num_workers=args.num_workers,
                                  collate_fn=collate_fn)
    # print("*"*80)
    # print(len(dataset.question_idxs))

    # for question_idx in dataset.question_idxs:
    #    print(question_idx)
    #    print("*" * 80)

    # print(self.question_idxs[question_idx])
    # self.question_idxs[idx]
    # print("data_loader: ",data_loader)
    # Evaluate
    log.info(f'Evaluating on {args.split} split...')
    nll_meter = util.AverageMeter()
    pred_dict = {}  # Predictions for TensorBoard
    sub_dict = {}  # Predictions for submission
    eval_file = vars(args)[f'{args.split}_eval_file']
    with open(eval_file, 'r') as fh:
        gold_dict = json_load(fh)

    # create statistics
    # print("*"*80)
    # print(len(gold_dict))
    # print(gold_dict['1']['question'])

    count_questions_type = defaultdict(lambda: 0)

    audit_trail_from_question_type = defaultdict(lambda: [])
    list_of_interrogative_pronouns = ["what", "whose", "why", "which", "where", "when", "how", "who", "whom"]

    for index in range(1, len(gold_dict)):
        # transform the question in lower case to simplify the analysis, thus losing the benefit of the capital letters
        # possibly indicating the position of the interrogative pronoun in the sentence.
        question_lower_case = gold_dict[str(index)]['question'].lower()

        list_question_lower_case_with_punctuation = question_lower_case.split()

        #
        question_lower_case = []
        for item in list_question_lower_case_with_punctuation:
            question_lower_case.append(item.translate({ord(i): None for i in ",.<>!@£$%^&*()_-+=?'"}))


        # defining a variable for the first word
        first_word_question_lower_case = question_lower_case[0]

        # defining variable for the second word
        second_word_question_lower_case = question_lower_case[1]

        # defining variable for the first and second word
        combined_first_and_second_words = first_word_question_lower_case + " " + second_word_question_lower_case

        #printing on the screen test for debugging purpose


        # Analyzing the sentence
        if first_word_question_lower_case in list_of_interrogative_pronouns:
            count_questions_type[first_word_question_lower_case] += 1
            audit_trail_from_question_type[first_word_question_lower_case].append(str(index))
        # composed question starting by in
        elif first_word_question_lower_case == "in":
            if second_word_question_lower_case in list_of_interrogative_pronouns and second_word_question_lower_case !="whose":
                count_questions_type[combined_first_and_second_words] += 1
                audit_trail_from_question_type[combined_first_and_second_words].append(str(index))
            else:
                pronoun = find_first_interrogative_pronoun(list_of_interrogative_pronouns, question_lower_case)
                count_questions_type[pronoun] += 1
                audit_trail_from_question_type[pronoun].append(str(index))

        # composed question starting by by
        elif first_word_question_lower_case == "by":
            if second_word_question_lower_case in list_of_interrogative_pronouns \
                    and second_word_question_lower_case !="whom"\
                    and second_word_question_lower_case !="which"\
                    and second_word_question_lower_case !="when"\
                    and second_word_question_lower_case !="how":
                count_questions_type[combined_first_and_second_words] += 1
                audit_trail_from_question_type[combined_first_and_second_words].append(str(index))
            else:
                pronoun = find_first_interrogative_pronoun(list_of_interrogative_pronouns, question_lower_case)
                count_questions_type[pronoun] += 1
                audit_trail_from_question_type[pronoun].append(str(index))

        else:
            pronoun = find_first_interrogative_pronoun(list_of_interrogative_pronouns, question_lower_case)
            #if pronoun =="":
            #    print(">>", question_lower_case)
            count_questions_type[pronoun] += 1
            audit_trail_from_question_type[pronoun].append(str(index))
            # if pronoun =="":
            #    print(">>", question_lower_case.split())
            # print()
            #if first_word_question_lower_case == "if":
            #    print(">>", question_lower_case.split())

    # print(count_questions_type)
    # if gold_dict[str(index)]['question'].lower().split()[0] == "in":
    #    print(gold_dict[str(index)]['question'])

    reverse_dict_by_value = OrderedDict(sorted(count_questions_type.items(), key=lambda x: x[1]))
    # print(count_questions_type)
    # print(reverse_dict)
    #for k, v in reverse_dict_by_value.items():
    #   print( "%s: %s" % (k, v))
    print(audit_trail_from_question_type)
    # exit()
    with torch.no_grad(), \
         tqdm(total=len(dataset)) as progress_bar:
        for cw_idxs, cc_idxs, qw_idxs, qc_idxs, cw_pos, cw_ner, cw_freq, cqw_extra, y1, y2, ids in data_loader:
            # Setup for forward
            cw_idxs = cw_idxs.to(device)
            qw_idxs = qw_idxs.to(device)
            batch_size = cw_idxs.size(0)

            # Forward
            if args.model == 'bidaf':
                log_p1, log_p2 = model(cw_idxs, qw_idxs)
            else:
                log_p1, log_p2 = model(cw_idxs, qw_idxs, cw_pos, cw_ner, cw_freq, cqw_extra)

            y1, y2 = y1.to(device), y2.to(device)
            loss = F.nll_loss(log_p1, y1) + F.nll_loss(log_p2, y2)
            nll_meter.update(loss.item(), batch_size)

            # Get F1 and EM scores
            p1, p2 = log_p1.exp(), log_p2.exp()
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
