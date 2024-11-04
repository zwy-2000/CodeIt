# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import re

import datasets

from codeit.dsl.dsl import *
from codeit.utils import get_grid_size
from codeit.policy.Julian_tokenization.tokenization_functions import map_to_t5_token


def sparse_grid_text_encoder(grid):
    grid_size = get_grid_size(grid)
    colour_indices = {}
    try:
        background_colour = mostcolor(grid)
    except:
        if all(not x for x in grid):
            return "empty"
    colours = set(palette(grid))
    colours.remove(background_colour)
    for colour in colours:
        indices = set(ofcolor(grid, colour))
        integer = [f"{i},{j}" for i, j in indices]
        colour_indices[colour] = " ".join(integer)
    grid_text = f"{grid_size[0]}x{grid_size[1]} background={background_colour} "
    for colour, indices in colour_indices.items():
        grid_text += f"{colour}={indices} "
    return grid_text.strip()


### new
def sparse_grid_text_encoder_2(grid):
    grid_size = get_grid_size(grid)
    colour_indices = {}
    try:
        background_colour = mostcolor(grid)
    except:
        if all(not x for x in grid):
            return "empty"
    colours = set(palette(grid))
    colours.remove(background_colour)
    for colour in colours:
        indices = set(ofcolor(grid, colour))
        integer = [f"{i},{j}" for i, j in indices]
        colour_indices[colour] = " ".join(integer)
    background_colour = bgcolor_text(background_color=background_colour)
    grid_text = f"{grid_size[0]}x{grid_size[1]} bg={background_colour} "
    for colour, indices in colour_indices.items():
        colour = bgcolor_text(colour)
        grid_text += f"{colour}={indices} "
    return grid_text.strip()



def sparse_grid_text_decoder(grid_text):
    def parse_string(input_str):
        grid_size_pattern = r"(\d+)x(\d+)"
        background_pattern = r"background=(\d+)"
        colours_pattern = r"(\d+)\s*=\s*((?:\d+,\d+\s*)+(?=\s*\d+\s*=|\s*$))"
        grid_size_match = re.search(grid_size_pattern, input_str)
        background_match = re.search(background_pattern, input_str)
        colours_matches = re.findall(colours_pattern, input_str)
        result = {}
        result["grid_size"] = (int(grid_size_match.group(1)), int(grid_size_match.group(2)))
        result["background"] = int(background_match.group(1))
        result["colours"] = {}
        if colours_matches:
            for match in colours_matches:
                number, coordinates = match
                coords = re.findall(r"(\d+,\d+)", coordinates)
                coords_tuples = [(int(x.split(",")[0]), int(x.split(",")[1])) for x in coords]
                key = int(number)
                result["colours"][key] = coords_tuples
        return result

    parsed_grid_text = parse_string(grid_text)
    grid = tuple(
        [
            tuple([parsed_grid_text["background"] for i in range(parsed_grid_text["grid_size"][0])])
            for i in range(parsed_grid_text["grid_size"][1])
        ]
    )
    colour_indices = {}
    for colour in parsed_grid_text["colours"]:
        colour_indices[colour] = frozenset(
            [number for number in parsed_grid_text["colours"][colour]]
        )
    for colour, indices in colour_indices.items():
        grid = fill(grid, colour, indices)
    return grid


def create_dataset(tasks, n_examples, sparse=True, text_encoder=None):
    # data = {"task_id": [], "program": [], "initial_states": [], "terminal_states": []}
    data = {"task_id": [], "program": [], "sparse_task": []}
    for task in tasks.values():
        # entry = create_dataset_entry(task, n_examples, sparse=sparse)
        entry = create_dataset_entry_2(task, n_examples, sparse=sparse)
        data["task_id"].append(entry["task_id"])
        data["program"].append(entry["program"])
        # data["initial_states"].append(entry["initial_states"]) ########## where the problem happens
        # data["terminal_states"].append(entry["terminal_states"])
        data["sparse_task"].append(entry["sparse_task"])
        ##################################################
        # print("entry program in create_dataset",entry["program"]) # from this step the program is empty
        ##################################################
    data = datasets.Dataset.from_dict(data)
    return data


def create_dataset_entry(task, n_examples, sparse=True):
    if sparse:
        initial_states = "|".join(
            [
                sparse_grid_text_encoder(example["input"])
                for example in task.training_examples[:n_examples]
            ]
        )
        terminal_states = "|".join(
            [
                sparse_grid_text_encoder(example["output"])
                for example in task.training_examples[:n_examples]
            ]
        )
    else:
        initial_states = "|".join(
            [
                grid_to_colored_text(example["input"])
                for example in task.training_examples[:n_examples]
            ]
        )
        terminal_states = "|".join(
            [
                grid_to_colored_text(example["output"])
                for example in task.training_examples[:n_examples]
            ]
        )
    return {
        "task_id": task.task_key,
        "program": task.program_lines,
        "initial_states": initial_states,
        "terminal_states": terminal_states,
    }


### new
def create_dataset_entry_2(task, n_examples, sparse=True):
    if sparse:
        # initial_states = "|".join(
        #     [
        #         sparse_grid_text_encoder_2(example["input"])
        #         for example in task.training_examples[:n_examples]
        #     ]
        # )
        # terminal_states = "|".join(
        #     [
        #         sparse_grid_text_encoder_2(example["output"])
        #         for example in task.training_examples[:n_examples]
        #     ]
        # )
        sparse_task = ""
        for example in task.training_examples[:n_examples]:
            sparse_task += "new "+ sparse_grid_text_encoder_2(example["input"])+'|'+sparse_grid_text_encoder_2(example["output"])
    else:
        initial_states = "|".join(
            [
                grid_to_colored_text(example["input"])
                for example in task.training_examples[:n_examples]
            ]
        )
        terminal_states = "|".join(
            [
                grid_to_colored_text(example["output"])
                for example in task.training_examples[:n_examples]
            ]
        )
    return {
        "task_id": task.task_key,
        "program": task.program_lines,
        # "initial_states": initial_states,
        # "terminal_states": terminal_states,
        "sparse_task": sparse_task
    }


def tokenize_inputs(dataset_entry, tokenizer, input_state_max):
    input_ids = (
        tokenizer.encode(dataset_entry["initial_states"])[:-1][:input_state_max]
        + tokenizer.encode("\n", add_special_tokens=False)
        + tokenizer.encode(dataset_entry["terminal_states"], add_special_tokens=False)[
            : (input_state_max - 1)
        ]
    )
    return input_ids



### new
def tokenize_inputs_2(dataset_entry, tokenizer, input_state_max):
    input_ids = (
        tokenizer.encode(dataset_entry["sparse_task"], add_special_tokens=False)[
            : (input_state_max*4)
        ]
    )
    return input_ids



def tokenize_simple_seq_2_seq(dataset_entry, tokenizer, input_state_max, max_tokens):
    example = {}
    # example["input_ids"] = tokenize_inputs(dataset_entry, tokenizer, input_state_max)
    example["input_ids"] = tokenize_inputs_2(dataset_entry, tokenizer, input_state_max)    
    example["attention_mask"] = [1] * len(example["input_ids"])
    ##### this is where the tokenization of program happens
    # example["labels"] = tokenizer.encode(dataset_entry["program"], add_special_tokens=True)[
    #     :max_tokens
    # ]
    example["labels"] = Julian_mapping(dataset_entry["program"], tokenizer)
    ###############
    # x1 = tokenizer.encode(dataset_entry["program"], add_special_tokens=True)[
    #     :max_tokens
    # ]
    # print('original method:',tokenizer.decode(x1))
    # print('Julian method', tokenizer.decode(example["labels"])) ########this is important#########
    ###############

    example["task_id"] = tokenizer.encode(dataset_entry["task_id"], add_special_tokens=False)[
        :max_tokens
    ]
    # print('decoded:', tokenizer.decode(example["labels"]))
    # print('program: ',example["labels"])
    # if example["labels"] == []:
    #     print("_________empty labels____________") ################################
    #     print(dataset_entry) ## by this step, the entry program is already empty ('')


    return example


def tokenize_task(
    task, tokenizer, n_examples, input_state_max, max_tokens, sparse=True, text_encoder=None
):
    # entry = create_dataset_entry(task, n_examples=n_examples, sparse=sparse)
    entry = create_dataset_entry_2(task, n_examples=n_examples, sparse=sparse) ## updated tokenization
    return tokenize_simple_seq_2_seq(
        tokenizer=tokenizer,
        dataset_entry=entry,
        input_state_max=input_state_max,
        max_tokens=max_tokens,
    )


class TextEncoder:
    def encode_grid(self, grid):
        # {'Grid':(1,0,1)} > {'Grid':background ...}
        return sparse_grid_text_encoder(grid)

    def decode_grid(self, state_text):
        # background > text
        return sparse_grid_text_decoder(state_text)


def grid_to_colored_text(grid):
    color_map = {
        0: "red",
        1: "green",
        2: "blue",
        3: "yellow",
        4: "orange",
        5: "purple",
        6: "cyan",
        7: "magenta",
        8: "brown",
        9: "black",
    }
    return "\n".join("".join(color_map[num] for num in row) for row in grid)


def bgcolor_text(background_color):
    color_map = {
        0: "\u2581Black",
        1: "\u2581Blue",
        2: "\u2581Red",
        3: "\u2581Green",
        4: "\u2581Yellow",
        5: "\u2581Gray",
        6: "\u2581Purple",
        7: "\u2581Orange",
        8: "\u2581Azure",
        9: "\u2581Brown",
        10: "\u2581White"
    }
    # new_bg = []
    # for color in background_color:
    #     new_bg.append(color_map[color])
    if background_color in color_map:
        new_bg = color_map[background_color]
    else:
        new_bg = str(background_color)
    return new_bg


######################### functions for mapping programs #########################

def Julian_mapping(target_string, tokenizer):
    target_token,_ = map_to_t5_token(target_string, extra_token= ['sym_aft_func', 'BoF', 'EoF'], tokenizer=tokenizer,
                                         loading_new_mappings=False, path_to_mapping='codeit/policy/Julian_tokenization/dsl_token_mappings_T5.json')
    ## add T5tokenization of the target token
    target_token_ids = tokenizer.convert_tokens_to_ids(target_token)+[1]
    return target_token_ids