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
import copy
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import matplotlib
import random

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


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", ignore=None, **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    # print("data: ", data)
    # print("type(data): ", type(data))
    original_data = copy.deepcopy(data)
    # print("original_data: ", original_data)
    # print("type(original_data): ", type(original_data))
    average_to_replace_NA = np.array([i for i in original_data.flatten() if i != "NA"]).astype(np.float).mean()

    data[data == "NA"] = average_to_replace_NA
    data = np.float32(data)
    # print("After removing NA")
    # print("data: ", data)
    # print("type(data): ", type(data))

    # data = data.tolist()

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.1f}",
                     textcolors=["black", "white"],
                     threshold=None, ignore=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """
    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 12

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # deal with the case of absent value:
    copied_data = copy.deepcopy(data)
    print("copied_data: ", copied_data)

    # print("original_data: ", original_data)
    # print("type(original_data): ", type(original_data))
    average_to_replace_NA = np.array([i for i in copied_data.flatten() if i != "NA" and i != ""]).astype(
        np.float).mean()

    data[data == "NA"] = average_to_replace_NA
    data[data == ""] = average_to_replace_NA
    data[data == 0] = average_to_replace_NA
    data = np.float32(data)

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    plt.rc('axes', titlesize=7)  # fontsize of the axes title
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if ignore != None:
                if (i, j) not in ignore:
                    print("(i,j):", (i, j), " is in ignore: ", (i, j) not in ignore)
                    kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
                    text = im.axes.text(j, i, valfmt(data[i, j], None), **kw, fontsize=7)
                    texts.append(text)
            else:
                kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
                text = im.axes.text(j, i, valfmt(data[i, j], None), **kw, fontsize=7)
                texts.append(text)

    return texts


def main(args, actions = None):
    """"
    actions is a tuple (action, number of actions to be taken)

    action can be either: "substitute", "delete" or "add".
    number of actions to be taken: the number of words to apply the "substitute", "delete" or "add" action.

    """

    # check that actions parameters received
    #print("actions: ",actions)



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
    #print(len(dataset.question_idxs))

    #for question_idx in dataset.question_idxs:
    #    print(question_idx)
    #    print("*" * 80)

    #print(self.question_idxs[question_idx])
    #self.question_idxs[idx]
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

        list_question_lower_case_with_punctuation = question_lower_case.translate({ord(i): " " for i in "'"}).split()

        #
        question_lower_case = []
        for item in list_question_lower_case_with_punctuation:
            question_lower_case.append(item.translate({ord(i): "" for i in ",.<>!@£$%^&*()_-+=?"}))

        # defining a variable for the first word
        first_word_question_lower_case = question_lower_case[0]

        # defining variable for the second word
        second_word_question_lower_case = question_lower_case[1]

        # defining variable for the first and second word
        combined_first_and_second_words = first_word_question_lower_case + " " + second_word_question_lower_case

        # printing on the screen test for debugging purpose

        # Analyzing the sentence
        if first_word_question_lower_case in list_of_interrogative_pronouns:
            count_questions_type[first_word_question_lower_case] += 1
            audit_trail_from_question_type[first_word_question_lower_case].append(str(index))
        # composed question starting by in
        elif first_word_question_lower_case == "in":
            if second_word_question_lower_case in list_of_interrogative_pronouns and second_word_question_lower_case != "whose":
                count_questions_type[combined_first_and_second_words] += 1
                audit_trail_from_question_type[combined_first_and_second_words].append(str(index))
            else:
                pronoun = find_first_interrogative_pronoun(list_of_interrogative_pronouns, question_lower_case)
                count_questions_type[pronoun] += 1
                audit_trail_from_question_type[pronoun].append(str(index))

        # composed question starting by by
        elif first_word_question_lower_case == "by":
            if second_word_question_lower_case in list_of_interrogative_pronouns \
                    and second_word_question_lower_case != "whom" \
                    and second_word_question_lower_case != "which" \
                    and second_word_question_lower_case != "when" \
                    and second_word_question_lower_case != "how":
                count_questions_type[combined_first_and_second_words] += 1
                audit_trail_from_question_type[combined_first_and_second_words].append(str(index))
            else:
                pronoun = find_first_interrogative_pronoun(list_of_interrogative_pronouns, question_lower_case)
                count_questions_type[pronoun] += 1
                audit_trail_from_question_type[pronoun].append(str(index))

        else:
            pronoun = find_first_interrogative_pronoun(list_of_interrogative_pronouns, question_lower_case)
            # if pronoun =="":
            #    print(">>", question_lower_case)
            #    print("@@@", gold_dict[str(index)]['question'])
            count_questions_type[pronoun] += 1
            audit_trail_from_question_type[pronoun].append(str(index))
            # if pronoun =="":
            #    print(">>", question_lower_case.split())
            # print()
            # if first_word_question_lower_case == "if":
            #    print(">>", question_lower_case.split())

    # print(count_questions_type)
    # if gold_dict[str(index)]['question'].lower().split()[0] == "in":
    #    print(gold_dict[str(index)]['question'])

    reverse_dict_by_value = OrderedDict(sorted(count_questions_type.items(), key=lambda x: x[1]))
    # print(count_questions_type)
    total_questions = sum(count_questions_type.values())
    # print(reverse_dict)
    # for k, v in reverse_dict_by_value.items():
    #   print( "%s: %s and in percentage: %s" % (k, v, 100*v/total_questions))
    # print(audit_trail_from_question_type)
    # exit()
    with torch.no_grad(), \
         tqdm(total=len(dataset)) as progress_bar:
        for cw_idxs, cc_idxs, qw_idxs, qc_idxs, cw_pos, cw_ner, cw_freq, cqw_extra, y1, y2, ids in data_loader:
            # Setup for forward

            # **********************************************************
            #
            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            #
            # Where we make the modif:

            if actions[0] == "substitute":
                # substitute to random token in each question of the batch (substitution is made within the same sentence:
                # print("batch size: ", cw_idxs.size()[0])
                batch_size = cw_idxs.size()[0]
                number_of_actions = actions[1]
                for _ in range(number_of_actions):
                    length_index_batch = cw_idxs.size()[1]
                    #print("cw_idxs.size()[1] :", cw_idxs.size()[1])
                    for i in range(batch_size):
                        tensor_with_zero_value = ((cw_idxs[i] == 0).nonzero()).squeeze()
                        #print("value: ", cw_idxs[i])
                        #print(">>>", tensor_with_zero_value)
                        #print("torch.min(tensor_with_zero_value)): ",torch.min(tensor_with_zero_value))
                        #print("shape: ", tensor_with_zero_value.size())
                        #print("torch.min(tensor_with_zero_value)).item(): ", torch.min(tensor_with_zero_value)).item()

                        try:
                            first_zero_value = torch.min(tensor_with_zero_value)
                        except:
                            first_zero_value = length_index_batch



                        #if tensor_with_zero_value ==
                        #    (torch.min(tensor_with_zero_value))
                        #print("tensor: ", cw_idxs[i])
                        #print("item number: ", i,  " index of first zero value: ", ((cw_idxs[i] == 0).nonzero()).squeeze()[1])
                        if first_zero_value > 2:
                            select_item_idx_1 = random.randint(0, first_zero_value-1)
                            select_item_idx_2 = random.randint(0, first_zero_value-1)
                            #print("select_item_idx_1 before switch", select_item_idx_1, " value: ", cw_idxs[i, select_item_idx_1])
                            #print("select_item_idx_2  before switch", select_item_idx_2, " value: ", cw_idxs[i, select_item_idx_2])
                            save_value_1 = copy.deepcopy(cw_idxs[i, select_item_idx_1])
                            cw_idxs[i, select_item_idx_1] = cw_idxs[i, select_item_idx_2]
                            cw_idxs[i, select_item_idx_2] = save_value_1
                            #print("select_item_idx_1 after switch", select_item_idx_1, " value: ", cw_idxs[i, select_item_idx_1])
                            #print("select_item_idx_2  after switch", select_item_idx_2, " value: ", cw_idxs[i, select_item_idx_2])
                            #print("tensor: ", cw_idxs[i])

                    # print("length of question in batch :", length_index_batch)
                    # for batch in cw_idxs.size()[0]:
                    #    select_item_idx_1 = randint(0, length_index_batch-1)
                    #    select_item_idx_2 = randint(0, length_index_batch-1)
                        #print("select_item_idx_1", select_item_idx_1)
                        #print("select_item_idx_2", select_item_idx_2)

            elif actions[0] == "delete":
                # substitute to random token in each question of the batch (substitution is made within the same sentence:
                # print("batch size: ", cw_idxs.size()[0])
                batch_size = cw_idxs.size()[0]
                number_of_actions = actions[1]
                for _ in range(number_of_actions):
                    length_index_batch = cw_idxs.size()[1]
                    #print("cw_idxs.size()[1] :", cw_idxs.size()[1])
                    for i in range(batch_size):
                        tensor_with_zero_value = ((cw_idxs[i] == 0).nonzero()).squeeze()
                        #print("value: ", cw_idxs[i])
                        #print(">>>", tensor_with_zero_value)
                        #print("torch.min(tensor_with_zero_value)): ",torch.min(tensor_with_zero_value))
                        #print("shape: ", tensor_with_zero_value.size())
                        #print("torch.min(tensor_with_zero_value)).item(): ", torch.min(tensor_with_zero_value)).item()

                        try:
                            first_zero_value = torch.min(tensor_with_zero_value)
                        except:
                            first_zero_value = length_index_batch



                        #if tensor_with_zero_value ==
                        #    (torch.min(tensor_with_zero_value))
                        #print("tensor: ", cw_idxs[i])
                        #print("item number: ", i,  " index of first zero value: ", ((cw_idxs[i] == 0).nonzero()).squeeze()[1])
                        if first_zero_value > 2:
                            select_item_idx_1 = random.randint(0, first_zero_value - 1)
                            #select_item_idx_2 = random.randint(0, first_zero_value-1)
                            #print("select_item_idx_1 before switch", select_item_idx_1, " value: ", cw_idxs[i, select_item_idx_1])
                            #print("select_item_idx_2  before switch", select_item_idx_2, " value: ", cw_idxs[i, select_item_idx_2])
                            #save_value_1 = copy.deepcopy(cw_idxs[i, select_item_idx_1])

                            #cw_idxs[i, select_item_idx_1] = random.randint(1, 50000)
                            cw_idxs[i, :] = torch.cat((cw_idxs[i,0:select_item_idx_1], cw_idxs[i,select_item_idx_1+1:],torch.tensor([0])), -1)
                            #new_tensor[:] = torch.cat((tensor[0:2], tensor[3:],torch.tensor([0])), 0)

                            # cw_idxs[i, select_item_idx_2] = save_value_1
                            #print("select_item_idx_1 after switch", select_item_idx_1, " value: ", cw_idxs[i, select_item_idx_1])
                            #print("select_item_idx_2  after switch", select_item_idx_2, " value: ", cw_idxs[i, select_item_idx_2])
                            #print("tensor: ", cw_idxs[i])

                    # print("length of question in batch :", length_index_batch)
                    # for batch in cw_idxs.size()[0]:
                    #    select_item_idx_1 = randint(0, length_index_batch-1)
                    #    select_item_idx_2 = randint(0, length_index_batch-1)
                        #print("select_item_idx_1", select_item_idx_1)
                        #print("select_item_idx_2", select_item_idx_2)



            elif actions[0] == "add":
                # substitute to random token in each question of the batch (substitution is made within the same sentence:
                # print("batch size: ", cw_idxs.size()[0])
                batch_size = cw_idxs.size()[0]
                number_of_actions = actions[1]
                for _ in range(number_of_actions):
                    length_index_batch = cw_idxs.size()[1]
                    #print("cw_idxs.size()[1] :", cw_idxs.size()[1])
                    for i in range(batch_size):
                        tensor_with_zero_value = ((cw_idxs[i] == 0).nonzero()).squeeze()
                        #print("value: ", cw_idxs[i])
                        #print(">>>", tensor_with_zero_value)
                        #print("torch.min(tensor_with_zero_value)): ",torch.min(tensor_with_zero_value))
                        #print("shape: ", tensor_with_zero_value.size())
                        #print("torch.min(tensor_with_zero_value)).item(): ", torch.min(tensor_with_zero_value)).item()

                        try:
                            first_zero_value = torch.min(tensor_with_zero_value)
                        except:
                            first_zero_value = length_index_batch



                        #if tensor_with_zero_value ==
                        #    (torch.min(tensor_with_zero_value))
                        #print("tensor: ", cw_idxs[i])
                        #print("item number: ", i,  " index of first zero value: ", ((cw_idxs[i] == 0).nonzero()).squeeze()[1])
                        if first_zero_value > 2:
                            select_item_idx_1 = random.randint(0, first_zero_value - 1)
                            #select_item_idx_2 = random.randint(0, first_zero_value-1)
                            #print("select_item_idx_1 before switch", select_item_idx_1, " value: ", cw_idxs[i, select_item_idx_1])
                            #print("select_item_idx_2  before switch", select_item_idx_2, " value: ", cw_idxs[i, select_item_idx_2])
                            #save_value_1 = copy.deepcopy(cw_idxs[i, select_item_idx_1])
                            cw_idxs[i, select_item_idx_1] = random.randint(1, 50000)
                            # cw_idxs[i, select_item_idx_2] = save_value_1
                            #print("select_item_idx_1 after switch", select_item_idx_1, " value: ", cw_idxs[i, select_item_idx_1])
                            #print("select_item_idx_2  after switch", select_item_idx_2, " value: ", cw_idxs[i, select_item_idx_2])
                            #print("tensor: ", cw_idxs[i])

                    # print("length of question in batch :", length_index_batch)
                    # for batch in cw_idxs.size()[0]:
                    #    select_item_idx_1 = randint(0, length_index_batch-1)
                    #    select_item_idx_2 = randint(0, length_index_batch-1)
                        #print("select_item_idx_1", select_item_idx_1)
                        #print("select_item_idx_2", select_item_idx_2)

            elif actions[0] == "add2":
                # substitute to random token in each question of the batch (substitution is made within the same sentence:
                # print("batch size: ", cw_idxs.size()[0])
                batch_size = cw_idxs.size()[0]
                number_of_actions = actions[1]
                for _ in range(number_of_actions):
                    length_index_batch = cw_idxs.size()[1]
                    # print("cw_idxs.size()[1] :", cw_idxs.size()[1])
                    for i in range(batch_size):
                        tensor_with_zero_value = ((cw_idxs[i] == 0).nonzero()).squeeze()
                        # print("value: ", cw_idxs[i])
                        # print(">>>", tensor_with_zero_value)
                        # print("torch.min(tensor_with_zero_value)): ",torch.min(tensor_with_zero_value))
                        # print("shape: ", tensor_with_zero_value.size())
                        # print("torch.min(tensor_with_zero_value)).item(): ", torch.min(tensor_with_zero_value)).item()

                        try:
                            first_zero_value = torch.min(tensor_with_zero_value)
                        except:
                            first_zero_value = length_index_batch

                        # if tensor_with_zero_value ==
                        #    (torch.min(tensor_with_zero_value))
                        # print("tensor: ", cw_idxs[i])
                        # print("item number: ", i,  " index of first zero value: ", ((cw_idxs[i] == 0).nonzero()).squeeze()[1])
                        if first_zero_value > 2:
                            select_item_idx_1 = random.randint(0, first_zero_value - 1)
                            select_item_idx_2 = random.randint(0, first_zero_value - 1)
                            # select_item_idx_2 = random.randint(0, first_zero_value-1)
                            # print("select_item_idx_1 before switch", select_item_idx_1, " value: ", cw_idxs[i, select_item_idx_1])
                            # print("select_item_idx_2  before switch", select_item_idx_2, " value: ", cw_idxs[i, select_item_idx_2])
                            # save_value_1 = copy.deepcopy(cw_idxs[i, select_item_idx_1])
                            cw_idxs[i, select_item_idx_1] = random.randint(1, 50000)
                            cw_idxs[i, select_item_idx_2] = random.randint(1, 50000)
                            # cw_idxs[i, select_item_idx_2] = save_value_1
                            # print("select_item_idx_1 after switch", select_item_idx_1, " value: ", cw_idxs[i, select_item_idx_1])
                            # print("select_item_idx_2  after switch", select_item_idx_2, " value: ", cw_idxs[i, select_item_idx_2])
                            # print("tensor: ", cw_idxs[i])

                    # print("length of question in batch :", length_index_batch)
                    # for batch in cw_idxs.size()[0]:
                    #    select_item_idx_1 = randint(0, length_index_batch-1)
                    #    select_item_idx_2 = randint(0, length_index_batch-1)
                    # print("select_item_idx_1", select_item_idx_1)
                    # print("select_item_idx_2", select_item_idx_2)



            else:
                print("Incorrect command: exiting")
                exit()
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

        # Printing information for questions without interrogative pronouns
        """"
        print("len(gold_dict): ", len(gold_dict))
        print("len(pred_dict): ", len(pred_dict))
        print("Is gold_dict.keys() identical to pred_dict.keys(): ", gold_dict.keys()==pred_dict.keys())
        if gold_dict.keys()!=pred_dict.keys():
            for key in gold_dict.keys():
                if key not in pred_dict.keys():
                    print("key ", key, " missing in pred_dict.keys(")
        """
        results_list = [('NLL', nll_meter.avg),
                        ('F1', results['F1']),
                        ('EM', results['EM'])]
        if args.use_squad_v2:
            results_list.append(('AvNA', results['AvNA']))
        results = OrderedDict(results_list)

        # Computing the F1 score for each type of question
        print("for ", actions, ": ",results['F1'])

        # create a list of the types of questions by extracting the keys from the dict audit_trail_from_question_type

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

    return results['F1']
if __name__ == '__main__':

    #proposed_actions = ('add', 1)
    #print("result for ", proposed_actions[0], ": ", main(get_test_args(), actions = proposed_actions))


    max_steps = 20

    substitute_F1_values = []
    add_F1_values = []
    delete_F1_values = []


    #
    for nb in range(0,max_steps):
        proposed_actions = ('delete', nb)
        delete_F1_values.append(main(get_test_args(), actions = proposed_actions))

    #
    #
    #
    #     #proposed_actions = ('delete', nb)
    #     #delete_F1_values.append(main(get_test_args(), actions = proposed_actions))
    #
    # print("Computing add2")
    # for nb in range(0, max_steps):
    #     proposed_actions = ('add2', nb)
    #     add_F1_values.append(main(get_test_args(), actions=proposed_actions))
    #     print(">>>", nb, " : ", add_F1_values)
    # print("add OK")
    # print("result for ",proposed_actions[0], ": ", substitute_F1_values)
    #
    max_steps = 20
    x = np.linspace(0, max_steps, max_steps )
    # substitute_F1_values = [68.33486323626532, 67.71919483508022, 67.08127587293258, 66.42602368678533, 65.6553721649929, 65.23549199487181, 64.14933543655921, 64.0103589720414, 63.303466448581716, 62.618198893603875]
    substitute_F1_values = [68.33486323626532, 67.23285228638218, 66.73308354531125, 66.14306850142398, 65.54881628728191, 65.33088188555581,
     64.13253393487105, 63.578903655839724, 63.27890305924118, 62.8772705882559, 62.54967576285084, 62.2195203002712,
     61.49383994536334, 61.31659100527948, 60.49810802197565, 58.83797719121586, 59.1433537536782, 58.24023034655501,
     58.04349629609232, 58.00730678396045]

    add_F1_values = [68.33486323626532, 67.98018113756014, 67.54894054587062, 67.58598247051836, 66.9322413680687, 67.0811889078869,
     66.9425174656538, 66.57131296474456, 66.62162059512168, 66.21431805860604, 65.75180147983815, 65.38114705135808,
     65.28818677866228, 65.26705644496026, 64.50154623612728, 64.65359901744826, 64.51610661460624, 63.90690425608921,
     64.02092710461305, 64.1271014387607]

    add2_F1_values = [68.33486323626532, 67.54839065720493, 67.09879099264052, 66.81830361853085, 66.1604872901321, 65.21080053247125,
     64.80447065879143, 64.7936578323173, 64.7998150308031, 64.54904675195431, 64.01599496348219, 63.805717083336326,
     62.9345214996013, 62.32016947001721, 61.49306358215139, 61.47981575321356, 61.1429887294174, 61.582206379069255,
     61.05709145281228, 60.66102968134862]

    delete_F1_values = np.zeros(max_steps)
    # Initialise the figure and axes.
    fig, ax = plt.subplots(1, figsize=(8, 6))

    # Set the title for the figure
    #fig.suptitle('Influence of adversarial attacks on F1 score', fontsize=15)

    # Draw all the lines in the same plot, assigning a label for each one to be
    # shown in the legend.
    ax.plot(x, substitute_F1_values, color="red", label="substitute two values", marker='*')
    ax.plot(x, add_F1_values, color="green", label="add one value", marker='o')
    ax.plot(x, add2_F1_values, color="blue", label="add 2 values at each step", marker='+')
    # ax.plot(x, delete_F1_values, color="pink", label="delete", marker='x')

    # Add a legend, and position it on the lower right (with no box)
    plt.legend(loc="lower left", frameon=True)

    plt.show()
