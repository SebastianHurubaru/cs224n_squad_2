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
            cbar_kw={}, cbarlabel="", **kwargs):
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

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
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

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

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
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts




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
            #    print("@@@", gold_dict[str(index)]['question'])
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
    total_questions = sum(count_questions_type.values())
    # print(reverse_dict)
    #for k, v in reverse_dict_by_value.items():
    #   print( "%s: %s and in percentage: %s" % (k, v, 100*v/total_questions))
    #print(audit_trail_from_question_type)
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

        # extra-test by Francois
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
        #
        #    audit_trail_from_question_type[pronoun].append(str(index))

        # create a list of the types of questions by extracting the keys from the dict audit_trail_from_question_type
        types_of_questions = list(audit_trail_from_question_type.keys())

        gold_dict_per_type_of_questions = defaultdict(lambda: [])
        pred_dict_per_type_of_questions = {}

        gold_dict_per_type_of_questions_start = {}
        pred_dict_per_type_of_questions_start = {}

        gold_dict_per_type_of_questions_middle = {}
        pred_dict_per_type_of_questions_middle = {}

        gold_dict_per_type_of_questions_end = {}
        pred_dict_per_type_of_questions_end = {}

        for type_of_questions in types_of_questions:
            #gold_pred = {key: value for key, value in gold_dict.items() if key in audit_trail_from_question_type[type_of_questions]}
            #lst_pred = {key: value for key, value in pred_dict.items() if key in audit_trail_from_question_type[type_of_questions]}

            # Create two dictionnaries for each type of sentence for gold_dict_per_type_of_questions and pred_dict_per_type_of_questions
            gold_dict_per_type_of_questions[type_of_questions] = {key: value for key, value in gold_dict.items() if key in audit_trail_from_question_type[type_of_questions] and key in pred_dict.keys()}
            pred_dict_per_type_of_questions[type_of_questions] = {key: value for key, value in pred_dict.items() if key in audit_trail_from_question_type[type_of_questions] and key in pred_dict.keys()}
            print(type_of_questions," F1 score: ", util.eval_dicts(gold_dict_per_type_of_questions[type_of_questions], pred_dict_per_type_of_questions[type_of_questions], args.use_squad_v2)['F1'])

            gold_dict_per_type_of_questions_start[type_of_questions] = {key: value for key, value in gold_dict.items() if key in audit_trail_from_question_type[type_of_questions] and key in pred_dict.keys()}
            pred_dict_per_type_of_questions_start[type_of_questions] = {key: value for key, value in pred_dict.items() if key in audit_trail_from_question_type[type_of_questions] and key in pred_dict.keys()}

            gold_dict_per_type_of_questions_middle[type_of_questions] = {key: value for key, value in gold_dict.items() if key in audit_trail_from_question_type[type_of_questions] and key in pred_dict.keys()}
            pred_dict_per_type_of_questions_middle[type_of_questions] = {key: value for key, value in pred_dict.items() if key in audit_trail_from_question_type[type_of_questions] and key in pred_dict.keys()}

            gold_dict_per_type_of_questions_end[type_of_questions] = {key: value for key, value in gold_dict.items() if key in audit_trail_from_question_type[type_of_questions] and key in pred_dict.keys()}
            pred_dict_per_type_of_questions_end[type_of_questions] = {key: value for key, value in pred_dict.items() if key in audit_trail_from_question_type[type_of_questions] and key in pred_dict.keys()}


            for key, value in gold_dict.items():
                #if key in audit_trail_from_question_type[type_of_questions] and key in pred_dict.keys():
                if key in audit_trail_from_question_type[type_of_questions] and type_of_questions != "" and key in pred_dict_per_type_of_questions[type_of_questions]:
                    """
                    print("type_of_questions: ",type_of_questions)
                    print("key: ", key)
                    print("question: ", value["question"])
                    sub_index = value["question"].lower().find(type_of_questions)
                    print("sub_index: ",sub_index)
                    test_fc = value["question"].lower().find(type_of_questions)
                    print("present type of the var: ",type(test_fc))
                    #print("question: ", value["question"][str(key)])
                    print("length of the question: ", len(value["question"]))
                    print('Position of the interrogative pronoun in the question:', )
                    """
            # Create two dictionnaries for each type of sentence based at the start of the sentence

                    if value["question"].lower().find(type_of_questions) == 1 or value["question"].lower().find(type_of_questions) == 0:
                        #print("BEGINNING")
                        if type_of_questions != "":
                            try:
                                del gold_dict_per_type_of_questions_middle[type_of_questions][key]
                            except KeyError:
                                pass

                            try:
                                del pred_dict_per_type_of_questions_middle[type_of_questions][key]
                            except KeyError:
                                pass

                            try:
                                del gold_dict_per_type_of_questions_end[type_of_questions][key]
                            except KeyError:
                                pass

                            try:
                                del pred_dict_per_type_of_questions_end[type_of_questions][key]
                            except KeyError:
                                pass

                        #pred_dict_per_type_of_questions_start[type_of_questions] = {key: pred_dict[key] for key in
                        #                                                            gold_dict_per_type_of_questions_start[
                        #                                                                type_of_questions].keys()}
                    elif value["question"].lower().find(type_of_questions) >= len(value["question"])-len(type_of_questions)-5:
                        #print("END")

                        if type_of_questions != "":
                            try:
                                del gold_dict_per_type_of_questions_middle[type_of_questions][key]
                            except KeyError:
                                pass

                            try:
                                del pred_dict_per_type_of_questions_middle[type_of_questions][key]
                            except KeyError:
                                pass

                            try:
                                del gold_dict_per_type_of_questions_start[type_of_questions][key]
                            except KeyError:
                                pass

                            try:
                                del pred_dict_per_type_of_questions_start[type_of_questions][key]
                            except KeyError:
                                pass

                        #print("type_of_questions: ",type_of_questions)
                        #sub_index = value["question"].lower().find(type_of_questions)
                        #print("sub_index: ", sub_index)
                        #print("len(value['question']) - len(type_of_questions) - 2: ", len(value["question"])-len(type_of_questions)-2)
                        #start_string = len(value["question"])-len(type_of_questions)-6
                        #end_string = len(value["question"])-1
                        #print("extract at the end: ", value["question"][start_string:end_string])
                    else:
                        #print("MIDDLE")
                        if type_of_questions != "":
                            try:
                                del gold_dict_per_type_of_questions_start[type_of_questions][key]
                            except KeyError:
                                pass

                            try:
                                del pred_dict_per_type_of_questions_start[type_of_questions][key]
                            except KeyError:
                                pass

                            try:
                                del gold_dict_per_type_of_questions_end[type_of_questions][key]
                            except KeyError:
                                pass

                            try:
                                del pred_dict_per_type_of_questions_end[type_of_questions][key]
                            except KeyError:
                                pass
                            pass

            """
            if  type_of_questions != "":
                gold_dict_per_type_of_questions_start[type_of_questions] = {key: value for key, value in gold_dict.items() if (key in audit_trail_from_question_type[type_of_questions] \
                                                                        and (value["question"].lower().find(type_of_questions) <= 1) \
                                                                        and key in pred_dict_per_type_of_questions[type_of_questions]) }
            """

            """
                for key in gold_dict_per_type_of_questions_start[type_of_questions].keys():
                    print("key:: ", key )
                    print("type(key):: ", type(key) )
                            print("pred_dict[,key,] : ", pred_dict[key])
                print("@@@@@@@@@@@@@@@@@@@@@@@@")
                
                pred_dict_per_type_of_questions_start[type_of_questions] = {key: pred_dict[key] for key in gold_dict_per_type_of_questions_start[type_of_questions].keys()}

                #pred_dict_per_type_of_questions_start[type_of_questions] = {key: value for key, value in pred_dict.items() if key in list(gold_dict_per_type_of_questions_start[type_of_questions].keys()) }

                # Create two dictionnaries for each type of sentence based at the end of the sentence
                gold_dict_per_type_of_questions_end[type_of_questions] = {key: value for key, value in gold_dict.items() if key in audit_trail_from_question_type[type_of_questions] \
                                                                        and value["question"].lower().find(type_of_questions) >= len(value["question"])-len(type_of_questions)-2 \
                                                                        and key in pred_dict_per_type_of_questions[type_of_questions]}


                pred_dict_per_type_of_questions_end[type_of_questions] = {key: pred_dict[key] for key in list(gold_dict_per_type_of_questions_end[type_of_questions].keys())}

                #print("*"*80)
                # Create two dictionnaries for each type of sentence based at the middle of the sentencecount_questions_type
                gold_dict_per_type_of_questions_middle[type_of_questions] = {key: value for key, value in gold_dict.items() if key not in list(gold_dict_per_type_of_questions_start[type_of_questions].keys()) \
                                                                            and key not in list(gold_dict_per_type_of_questions_end[type_of_questions].keys())}

                pred_dict_per_type_of_questions_middle[type_of_questions] = {key: pred_dict[key] for key in list(gold_dict_per_type_of_questions_end[type_of_questions].keys())}
            else:
                gold_dict_per_type_of_questions_start[""] = gold_dict_per_type_of_questions[""]
                pred_dict_per_type_of_questions_start[""] = pred_dict_per_type_of_questions[""]
                gold_dict_per_type_of_questions_end[""] = gold_dict_per_type_of_questions[""]
                pred_dict_per_type_of_questions_end[""] = pred_dict_per_type_of_questions[""]
                gold_dict_per_type_of_questions_middle[""] = gold_dict_per_type_of_questions[""]
                pred_dict_per_type_of_questions_middle[""] = pred_dict_per_type_of_questions[""]
        """

        positions_in_question = ["beginning", "middle", "end"]

        print(type_of_questions," F1 score: ", util.eval_dicts(gold_dict_per_type_of_questions[type_of_questions], pred_dict_per_type_of_questions[type_of_questions], args.use_squad_v2)['F1'])

        list_beginning = [util.eval_dicts(gold_dict_per_type_of_questions_start[type_of_questions], pred_dict_per_type_of_questions_start[type_of_questions], args.use_squad_v2)['F1']  for type_of_questions in types_of_questions]
        list_middle = [util.eval_dicts(gold_dict_per_type_of_questions_middle[type_of_questions], pred_dict_per_type_of_questions_middle[type_of_questions], args.use_squad_v2)['F1']  for type_of_questions in types_of_questions]
        list_end = [util.eval_dicts(gold_dict_per_type_of_questions_end[type_of_questions], pred_dict_per_type_of_questions_end[type_of_questions], args.use_squad_v2)['F1']  for type_of_questions in types_of_questions]

        #for type_of_questions in types_of_questions:
        #    print("gold_dict_per_type_of_questions_start[type_of_questions]: ",gold_dict_per_type_of_questions_start[type_of_questions])
        #    print("pred_dict_per_type_of_questions[type_of_questions]: ",pred_dict_per_type_of_questions[type_of_questions])

        F1 = np.array([list_beginning,
                        list_middle,
                        list_end])


        #F1 = np.array([[0, 0, 0, 0, 0.1, 0, 0, 0, 0, 0, 0, 0, 0],
        #                    [0, 0, 0, 0, 0.1, 0, 0, 0, 0, 0, 0, 0, 0],
        #                    [0, 0, 0, 0, 0.1, 0, 0, 0, 0, 0, 0, 0, 0]])

        fig, ax = plt.subplots()

        types_of_questions[types_of_questions.index("")] = "Implicit question without interrogative pronoun"

        im, cbar = heatmap(F1, positions_in_question, types_of_questions, ax=ax,
                           cmap="YlGn", cbarlabel="F1 scores")
        texts = annotate_heatmap(im, valfmt="{x:.1f}")

        fig.tight_layout()
        plt.show()

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
