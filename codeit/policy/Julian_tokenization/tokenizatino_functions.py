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