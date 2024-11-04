import json
import re
from Levenshtein import distance as levenshtein_distance
import numpy as np



def map_to_t5_token(string_solver,extra_token, tokenizer, loading_new_mappings, path_to_mapping):
    string_print, list_of_tokens = reformat_dsl_code(string_solver, extra_token=extra_token)
    dsl_token_mappings = load_token_mappings(filename=path_to_mapping)
    if list_of_tokens is None:
        error_message = "The list of tokens is empty. Please check the reformat_dsl_code function."
        return error_message
    list_of_tokens_T5 = map_list(list_of_tokens, dsl_token_mappings, tokenizer)

    # if loading_new_mappings:
    #     print("--------------------- Example DSL ---------------------")
    #     print('---- DSL with our Tokens ----')
    #     idx_BoF = string_print.index('#BoF')
    #     idx_EoF = string_print.index('#EoF')
    #     print(string_print[idx_BoF:idx_EoF+4])

    #     print('---- DSL List Tokens ----')
    #     idx_BoF = list_of_tokens.index('#BoF')
    #     idx_EoF = list_of_tokens.index('#EoF')
    #     print(list_of_tokens[idx_BoF:idx_EoF + 1])


    #     print("------ T5 list -------")
    #     idx_BoF = list_of_tokens_T5.index(dsl_token_mappings['#BoF'])
    #     idx_EoF = list_of_tokens_T5.index(dsl_token_mappings['#EoF'])
    #     print(list_of_tokens_T5[idx_BoF:idx_EoF+1])

    return list_of_tokens_T5, dsl_token_mappings


def reformat_dsl_code(original_code, extra_token = None):
    lines = original_code.split('\n')
    new_code = []
    previous = []  # Track variables defined in previous lines
    for line in lines:
        if line.strip().startswith('x') and '=' in line:  # Process lines defining variables
            parts = line.split('=')
            var_name = parts[0].strip()
            expression = parts[1].strip().replace('(', ' ').replace(')', '').replace(',', ' ')

            if extra_token is not None and 'sym_aft_func' in extra_token:
                tokens = expression.split()
                tokens.insert(1, ';')
                expression = ' '.join(tokens)

            if extra_token is not None and 'underscore' in extra_token:
                tokens = expression.split()
                tokens[0] = '_' + tokens[0]
                expression = ' '.join(tokens)

            new_line = f'#newline {var_name} {expression}'

            if extra_token is not None and 'var_to_num' in extra_token:
                # Tokenize the variable name
                var_tokenized = tokenize_variable(var_name)
                # Tokenize all variables within the expression
                expression_tokenized = tokenize_expression(expression)
                new_line = f'#newline {var_tokenized} {expression_tokenized}'

            if not (extra_token is not None and 'prev' in extra_token):
                new_code.append(new_line)

            # ----------- Handle the case when 'prev' is in extra_token ------------ #
            else: # if prev is in extra_token
                tokens = expression.split()
                function = tokens[0]
                args = tokens[1:]
                modified_args = []
                for arg in args:
                    if arg in previous[:-1] and arg[0] =='x':  # Checks if the argument is at least two steps back
                        # if arg in previous[:-1]:  # Checks if the argument is at least two steps back
                        # , but then previous.append(var_name)

                        modified_args.append('prev ' + arg)
                    else:
                        modified_args.append(arg)
                if extra_token is not None and 'var_to_num' in extra_token:
                    modified_args_tokenized = tokenize_expression(' '.join(modified_args))
                    new_line = f'#newline {var_tokenized} {function} {modified_args_tokenized}'
                else:
                    new_line = f'#newline {var_name} {function} ' + ' '.join(modified_args)
                new_code.append(new_line)
                previous.append(arg)  # Remember this var as defined

        elif line.strip().startswith('O ='):  # Handle the output assignment line
            parts = line.split('=')
            expression = parts[1].strip().replace('(', ' ').replace(')', '').replace(',', ' ')

            if extra_token is not None and 'sym_aft_func' in extra_token:
                tokens = expression.split()
                tokens.insert(1, ';')
                expression = ' '.join(tokens)

            if extra_token is not None and 'var_to_num' in extra_token:
                expression_tokenized = tokenize_expression(expression)
                new_line = f'#newline O {expression_tokenized}'
            else:
                new_line = f'#newline O {expression}'
            new_code.append(new_line)

        # ----------- Handle the case when for the begining of the function is in extra_token ------------ #
        elif line.strip().startswith('def'):  #BoF begining of Function
            new_line = "#BoF"
            previous = [] # the previous_vars are reset because we are in a new function
            if extra_token is not None and 'BoF' in extra_token:
                new_code.append(new_line) # Function definition line

        elif line.find('return') != -1:  # Handle the return statement
            new_line = "#EoF"
            if extra_token is not None and 'EoF' in extra_token:
                new_code.append(new_line)

    string_print = '\n'.join(new_code)
    list_of_tokens = string_print.split()
    return string_print, list_of_tokens


def load_token_mappings(filename):
    """Loads token mappings from a JSON file, handling potential errors."""
    try:
        with open(filename, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print("No existing token mappings found.")
        return None


def map_list(list_of_tokens, dsl_token_mappings, tokenizer):
    T5_tokens_list = []
    for token in list_of_tokens:
        if token in dsl_token_mappings:
            T5_tokens_list.append(dsl_token_mappings[token])
        else:
            try:
                # get the closest token with levenshtein distance
                all_tokens = list(dsl_token_mappings.values())
                best_match, score = get_best_match_levenstein(token,all_tokens , [])

                print(f"Input list contains a token that is not in the mapping: {token}, set loading_new_mappings to True to generate a new mapping.")
                print(f"Best match for {token} is {best_match} with a score of {score}")
                T5_tokens_list.append(best_match)
            except:
                # if the token is not in the mapping and there is no best match
                # use a random token from the T5 vocabulary
                print(f"Input list contains a token that is not in the mapping: {token}, set loading_new_mappings to True to generate a new mapping.")
                print("Randomly selecting a token from the T5 vocabulary.")
                T5_tokens_list.append(list(tokenizer.get_vocab().keys())[np.random.randint(0, len(tokenizer.get_vocab().keys()))])
    return T5_tokens_list


def tokenize_variable(var_name):
    """Helper function to tokenize variable names with digit separation."""
    match = re.match(r"(x)(\d+)", var_name)
    if match:
        var_prefix, var_numbers = match.groups()
        # Create a space-separated string of the prefix and each digit
        digit_tokens = ' '.join(var_numbers)  # Splitting the number into separate digits
        return f"{var_prefix} {digit_tokens}"
    return var_name  # Return original if it does not match expected pattern


def tokenize_expression(expression):
    """Helper function to tokenize all variable references in an expression."""
    tokens = expression.split()
    new_tokens = []
    for token in tokens:
        if 'x' in token and re.match(r"x\d+", token):
            # If token is a variable, tokenize it
            new_tokens.append(tokenize_variable(token))
        else:
            # Otherwise, add it unchanged
            new_tokens.append(token)
    return ' '.join(new_tokens)

def get_best_match_levenstein(token, all_tokens, used_matches):
    distances = [(other, levenshtein_distance(token, other)) for other in all_tokens]
    distances.sort(key=lambda x: x[1])

    for match, dist in distances:
        if match not in used_matches:
            return match, dist
    return None, None


def decode_action_custom(gen_id, tokenizer):
    extra_tokens = [
    'sym_aft_func',
    'EoF',
    'BoF'
    ]
    one_sample_pred = []
    print(gen_id)
    for _id in gen_id[0]:
        token = tokenizer.convert_ids_to_tokens(int(_id))
        one_sample_pred.append(token)
    print(one_sample_pred)
    our_token_sample = map_back(one_sample_pred, 'codeit/policy/Julian_tokenization/dsl_token_mappings_T5.json')
    our_token_sample = small_trick(our_token_sample, extra_tokens)
    print(our_token_sample)
    our_token_sample=reconstruct_code(our_token_sample, 'codeit/policy/Julian_tokenization/dsl_token_mappings_T5.json')
    print(our_token_sample)

    return [our_token_sample]


def map_back(list_of_tokens, filename):
    T5_map = load_token_mappings(filename=filename)
    # use the dictionary to map back the tokens
    # switch the key and values
    T5_map = {v: k for k, v in T5_map.items()}
    original_tokens = []
    for token in list_of_tokens:
        if token in T5_map:
            original_tokens.append(T5_map[token])
        else:
            original_tokens.append(token)

    return original_tokens

def small_trick(our_token_sample, extra_tokens):
    if 'BoF' in extra_tokens and 'var_to_num' in extra_tokens:
        our_token_sample[0] = '#BoF'
        our_token_sample[1] = '#newline'
        our_token_sample[2] = 'x'
        our_token_sample[3] = '1'
    elif 'BoF' in extra_tokens:
        our_token_sample[0] = '#BoF'
        our_token_sample[1] = '#newline'
        our_token_sample[2] = 'x1'
    elif 'var_to_num' in extra_tokens:
        our_token_sample[0] = '#newline'
        our_token_sample[1] = 'x'
        our_token_sample[2] = '1'
    else:
        our_token_sample[0] = '#newline'
        our_token_sample[1] = 'x1'
    return our_token_sample

def reconstruct_code(token_list, path_to_mapping):
    tokens = token_list
    output = []
    current_function = []
    # search for #newline from the back of the list
    last_idx_newline = len(tokens) - tokens[::-1].index('#newline') - 1
    i = 0

    # make the function header with the name of the function
    # import_statement = "from dsl import * \nfrom constants import *\n"
    # output.append(import_statement)
    # function_header = f"\n\ndef solve_{name_idx}(I: Grid) -> Grid:"
    # current_function.append(function_header)
    while i < len(tokens):
        token = tokens[i]

        if token == '#newline' or i==0 or token == '<pad>':
            try:
                current_function.append('\n')  # Add newline

                # get the indexes of the beginning and the end of the line
                if i == last_idx_newline:
                    # for last line in the code, it is only one time true in the end
                    idx_end_line = tokens.index('#EoF', i + 1)
                else:
                    idx_end_line = tokens.index('#newline', i + 1)
                    if '#EoF' in tokens[i + 1:idx_end_line]:
                        idx_of_end_func = tokens.index('#EoF', i + 1)
                        if idx_end_line > idx_of_end_func:
                            idx_end_line = idx_of_end_func


                code_line = rebuild_the_line(tokens, i, idx_end_line, path_to_mapping)
                current_function.extend(code_line)
                i = idx_end_line  # Move index to next relevant token
            except:
                current_function.append('\n')
                current_function.append('    # there was an error in the code in this line')
                i += 1
        elif token == '#EoF':
            # current_function.append("\n    return O")
            output.append(''.join(current_function))
            current_function = []
            break # Move to next token

        else:
            current_function.append(f"\n    # generated {token}")
            i += 1  # Skip unrecognized tokens or handle other cases if needed

    if current_function:
        output.append(''.join(current_function))

    # Return a single string that concatenates all function definitions
    return ''.join(output)



def rebuild_the_line(tokens, idx_beginning, idx_end_line, path_to_mapping):
    # load mapping
    dsl_token_mappings = load_token_mappings_utils(filename=path_to_mapping) # tokens[idx_beginning:idx_end_line]
    idx_of_break = tokens.index(';', idx_beginning + 1, idx_end_line)
    function_name = tokens[idx_of_break - 1]
    sub_tokens = tokens[idx_beginning:idx_of_break]
    if sub_tokens.count('x') > 1 or (sub_tokens.count('x') + sub_tokens.count('O')) > 1:
        reversed_sub_tokens = sub_tokens[::-1]
        reverse_index = reversed_sub_tokens.index('x')
        idx_var_name = idx_of_break - reverse_index - 1
        function_name = ''.join(tokens[idx_var_name:idx_of_break])
        idx_var_name = idx_var_name + 1
    else:
        idx_var_name = idx_of_break

    arguments = tokens[idx_of_break + 1:idx_end_line]
    args = []
    unrecognized_tokens = []
    # get all the arguments which are the inputs to the function
    for1 = 0
    # there is a unrecognized tokens
    while for1 < len(arguments):
        if not arguments[for1] in dsl_token_mappings:
            unrecognized_tokens.append(arguments[for1])
            for1 += 1
        elif arguments[for1] == 'x':
            for2 = for1 + 1
            argument = 'x'
            while for2 < len(arguments):
                if arguments[for2] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                    argument += arguments[for2]
                    for2 += 1
                else:
                    break
            args.extend([argument])
            for1 = for2

        else:
            args.extend([arguments[for1]])
            for1 += 1

    if tokens[idx_beginning + 1] == 'O':
        var_name = 'O'
        code_line = f"    {var_name} = {function_name}({', '.join(args)})"
    else:
        var_name = ''.join(tokens[idx_beginning+1:idx_var_name-1])
        code_line = f"    {var_name} = {function_name}({', '.join(args)})"
    if unrecognized_tokens:
        comment = ', '.join(unrecognized_tokens)
        code_line = code_line + f"   # there was an unrecognized token in the code: " + comment
    return code_line


def load_token_mappings_utils(filename="dsl_token_mappings_T5.json"):
    """Loads token mappings from a JSON file, handling potential errors."""
    try:
        with open(filename, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print("No existing token mappings found.")
        return None