import numpy as np
import ast
from collections import defaultdict

###################### get dsl

def extract_function_names_from_file(file_path):
    with open(file_path, "r") as file:
        file_content = file.read()
    
    # Parse the file content into an AST
    tree = ast.parse(file_content)
    
    # List to hold function names
    function_names = []
    
    # Traverse the AST and find all function definitions
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            function_names.append(node.name)
    
    return function_names

# Example usage
dsl_file_path = "codeit/dsl/dsl.py"  # Path to your dsl.py file
dsl_functions = extract_function_names_from_file(dsl_file_path)



# adjacency matrix
num_functions = 162 # 160 functions + Input Output
dependence_graph = np.zeros((num_functions, num_functions), dtype=int)


# function names to indices
function_to_index = {func: i for i, func in enumerate(dsl_functions)}
function_to_index['Input'] = 160
function_to_index['Output'] = 161

################# get programs

# extract program definitions as strings
def extract_program_definitions(file_path):
    with open(file_path, "r") as file:
        file_content = file.read()
    
    # Parse the file content into an AST
    tree = ast.parse(file_content)
    
    # List to hold function definitions as strings
    function_strings = []
    
    # Traverse the AST and find all function definitions
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # Get the function name
            func_name = node.name
            
            # Extract the full source code of the function
            start_line = node.lineno - 1  # line numbers are 1-based
            end_line = node.end_lineno  # end_lineno is available in Python 3.8+
            
            # Get the full function definition as a string
            function_string = "\n".join(file_content.splitlines()[start_line:end_line])
            
            # Add the function definition string to the list
            function_strings.append(function_string)
    
    return function_strings

# Example usage
solvers_file_path = "codeit/dsl/solvers.py"  # Path to your solvers.py file
programs = extract_program_definitions(solvers_file_path)









# Helper function to extract function calls and their input/output arguments from an AST node
def extract_function_calls(node):
    calls = {}
    targets_seq = []
    input_functions = set()

    for child in ast.walk(node):
        if isinstance(child, ast.Assign):  # Handling assignments (e.g., x = func())
            if isinstance(child.value, ast.Call):
                if isinstance(child.value.func, ast.Name):  # Function call

                    for target in child.targets:
                        targets_seq.append(target.id)
                        calls[target.id] = {'func':child.value.func.id, 'args': [arg.id for arg in child.value.args]}
                    

                    func_name = child.value.func.id                    
                    

                    # Check if "I" is in the arguments
                    for arg in child.value.args:
                        if isinstance(arg, ast.Name) and arg.id == "I":
                            input_functions.add(func_name)

    return targets_seq, calls, input_functions

# Analyze a single program and extract function call sequences, inputs, and outputs
def analyze_program(program_code, function_to_index, dependece_graph):
    # Parse the program code into an AST
    tree = ast.parse(program_code)
    
    # Extract function call sequences along with input/output tracking
    targets_list, function_calls, input_functions = extract_function_calls(tree)
    # print("target list", targets_list)
    # print("calls:", function_calls)
    # print("input:", input_functions)

    for target in targets_list:
        target_call = function_calls[target]
        func_follows = target_call['func']

        if func_follows in targets_list:
            func_follows = function_calls[func_follows]['func']

        for arg in target_call['args']:
            if arg in targets_list:
                prev_func = function_calls[arg]['func']
                if prev_func in targets_list:
                    prev_func = function_calls[prev_func]['func']
                prev_indx = function_to_index[prev_func]
                follows_indx = function_to_index[func_follows]
                dependence_graph[prev_indx,follows_indx] += 1
        
        if target == 'O':
            func_indx = function_to_index[func_follows]
            dependece_graph[func_indx, 161] += 1
    
    for input_func in input_functions:
        if input_func in targets_list:
            input_func = function_calls[input_func]['func']
        func_indx = function_to_index[input_func]
        dependence_graph[160, func_indx] += 1
    
    


    
    # Build connectivity information
    # connectivity = defaultdict(list)
    # for i in range(len(function_calls) - 1):
    #     current_func = function_calls[i]
    #     next_func = function_calls[i + 1]
    #     connectivity[current_func].append(next_func)
    
    # return connectivity, input_functions, output_functions
    return dependece_graph




for program_code in programs:
    # Analyze the example program
    # print('---------')
    # print(program_code)
    dependence_graph = analyze_program(program_code, function_to_index, dependence_graph)

    # for func, follows in connectivity.items():
    #     func_index = function_to_index[func]
    #     print(follows)
    #     for follow in follows:
    #         if follow == 'x2':
    #             print(program_code)
    #         follow_index = function_to_index[follow]
    #         dependence_graph[func_index,follow_index] += 1
    # for func in input_functions:
    #     func_index = function_to_index[func]
    #     dependence_graph[160,func_index] += 1
    # for func in output_functions:
    #     func_index = function_to_index[func]
    #     dependence_graph[func_index, 161] += 1


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df_denpendence_graph = pd.DataFrame(dependence_graph, index = function_to_index, columns= function_to_index)
result = df_denpendence_graph.to_json(r"mutate_weights/dependence_graph.json",orient='split')
df_denpendence_graph = pd.read_json('mutate_weights/dependence_graph.json', orient='split')



plt.figure(figsize = (40,30))

sns.heatmap(df_denpendence_graph, cmap = 'coolwarm')
plt.xlabel("next function", fontdict= {'size':70})
plt.ylabel("previous function", fontdict= {'size':70})

print('heatmap saved')
plt.savefig('mutate_weights/heatmap.png')


from json import loads, dumps


