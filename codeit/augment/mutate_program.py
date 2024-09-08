# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import ast
import copy
import random
import traceback
import pandas as pd

from codeit.augment.mutate_grid import valid_grid
from codeit.augment.type_inference import (
    CONSTANT_TO_TYPE_MAPPING,
    TypeInferer,
    display_type,
)
from codeit.dsl.arc_types import *
from codeit.dsl.dsl import *
from codeit.dsl.primitives import *


class ProgramMutator:
    def __init__(
        self,
        program,
        type_inferer,
        phi_var,
        phi_func,
        phi_arg,
        primitive_function_to_general_type_mapping,
        primitive_function_to_base_type_mapping,
        primitive_constant_to_type_mapping,
        general_type_to_primitive_function_mapping,
        base_type_to_primitive_function_mapping,
        type_to_primitive_constant_mapping,
    ):
        self.program = program
        self.program_ast = ast.parse(program)
        self.type_inferer = type_inferer
        self.phi_var = phi_var
        self.phi_func = phi_func
        self.phi_arg = phi_arg
        self.primitive_function_to_general_type_mapping = primitive_function_to_general_type_mapping
        self.primitive_function_to_base_type_mapping = primitive_function_to_base_type_mapping
        self.primitive_constant_to_type_mapping = primitive_constant_to_type_mapping
        self.general_type_to_primitive_function_mapping = general_type_to_primitive_function_mapping
        self.base_type_to_primitive_function_mapping = base_type_to_primitive_function_mapping
        self.type_to_primitive_constant_mapping = type_to_primitive_constant_mapping
        self.memory_index = None
        self.type_inferer.infer_type_from_ast(self.program_ast)
        ##
        self.dependence_weights = pd.read_json('mutate_weights/dependence_graph.json', orient='split')


    def mutate(self):
        mutation_choice = random.choices(
            ["replace_argument", "replace_function", "replace_variable"],
            weights=[self.phi_arg, self.phi_func, 1 - (self.phi_func + self.phi_arg)],
        )[0]
        assignments = [node for node in ast.walk(self.program_ast) if isinstance(node, ast.Assign)]
        # print("assignments:",assignments)
        # print("length of assignments:",len(assignments))
        node_to_mutate = random.choice(assignments)
        self.memory_index = assignments.index(node_to_mutate) + 1
        if mutation_choice == "replace_argument":
            arg_to_replace_id = random.choice(range(len(node_to_mutate.value.args)))
            arg_to_replace = node_to_mutate.value.args[arg_to_replace_id].id
            # print("arg_to_replace:", arg_to_replace)
            new_arg, new_variable_mutation = self.replace_argument(arg_to_replace=arg_to_replace)
            assignments[self.memory_index - 1].value.args[arg_to_replace_id].id = new_arg
            if new_variable_mutation:
                mutation = ("arg", arg_to_replace, new_arg, new_variable_mutation)
            else:
                mutation = ("arg", arg_to_replace, new_arg)
        elif mutation_choice == "replace_function":
            function_to_replace = node_to_mutate.value.func.id
            # print("function_to_replace:", function_to_replace)
            new_function = self.replace_function(function_to_replace=function_to_replace)
            assignments[self.memory_index - 1].value.func.id = new_function
            mutation = ("func", function_to_replace, new_function)
        else:
            variable_to_replace = node_to_mutate.targets[0].id
            # print("variable_to_replace:", variable_to_replace)
            new_variable_value = self.replace_variable(variable_to_replace=variable_to_replace)
            mutation = (
                "var_def",
                f"{node_to_mutate.value.func.id}({', '.join([arg.id for arg in node_to_mutate.value.args])})",
                new_variable_value,
            )
            new_node = ast.parse(new_variable_value).body[0].value
            assignments[self.memory_index - 1].value = new_node
        # print("mutation:", mutation)
        return mutation
    
    def mutate2(self):
        # mutation_node_weights = [0.27067, 0.27067, 0.18045, 0.09022, 0.03609] # Poisson lambda = 2
        mutation_node_weights = [0.36788, 0.18394, 0.06131] # Poisson lambda = 1

        assignments = [node for node in ast.walk(self.program_ast) if isinstance(node, ast.Assign)]
        if len(assignments) < 4:
            n_nodes = 1
        else:
            n_nodes = 3

        if n_nodes > 1:
            # n_weights = [w / sum(mutation_node_weights[:n_nodes]) for w in mutation_node_weights[:n_nodes]]
            n_mutation = random.choices(range(1,n_nodes+1), weights = mutation_node_weights)[0]
        else:
            n_mutation = 1
        # print("---------------len assign:", len(assignments))
        # print("---------------n mutation:",n_mutation)
        
        mutation = []


        nodes_to_mutate = random.sample(assignments, k = n_mutation)
        for node_to_mutate in nodes_to_mutate:
            mutation_choice = random.choices(
                ["replace_argument", "replace_function", "replace_variable"],
                weights=[self.phi_arg, self.phi_func, 1 - (self.phi_func + self.phi_arg)],
            )[0]
            # print("mutation_choice:", mutation_choice)
            self.memory_index = assignments.index(node_to_mutate) + 1
            if mutation_choice == "replace_argument":
                arg_to_replace_id = random.choice(range(len(node_to_mutate.value.args)))
                arg_to_replace = node_to_mutate.value.args[arg_to_replace_id].id
                # print("arg_to_replace:", arg_to_replace)
                new_arg, new_variable_mutation = self.replace_argument(arg_to_replace=arg_to_replace)
                assignments[self.memory_index - 1].value.args[arg_to_replace_id].id = new_arg
                if new_variable_mutation:
                    mutation.append(("arg", arg_to_replace, new_arg, new_variable_mutation)) ##
                else:
                    mutation.append(("arg", arg_to_replace, new_arg)) ##
                # print("mutation:", mutation)
            elif mutation_choice == "replace_function":
                function_to_replace = node_to_mutate.value.func.id
                # print("function_to_replace:", function_to_replace)
                new_function = self.replace_function(function_to_replace=function_to_replace)
                assignments[self.memory_index - 1].value.func.id = new_function
                mutation.append(("func", function_to_replace, new_function)) ##
                # print("mutation:", mutation)
            else:
                variable_to_replace = node_to_mutate.targets[0].id
                new_variable_value = self.replace_variable(variable_to_replace=variable_to_replace)
                mutation.append( (
                    "var_def",
                    f"{node_to_mutate.value.func.id}({', '.join([arg.id for arg in node_to_mutate.value.args])})",
                    new_variable_value,
                ))
                new_node = ast.parse(new_variable_value).body[0].value
                assignments[self.memory_index - 1].value = new_node
                # print("mutation:", mutation)
        # print('mutation', mutation)
        # print("program:", self.program)
        # print("program_ast:", ast.unparse(self.program_ast))
        return mutation
    
    def sample_two_func_amplified(self, desired_input_types, desired_output_type, existing_types): # *****
        dependence_weights = self.dependence_weights

        # dsl_list = list(dependence_weights.columns)
        # function_to_indx = {func: index for index, func in enumerate(dsl_list)}

        available_types = existing_types + list(self.type_to_primitive_constant_mapping.keys())
        # print("available types:", available_types)

        # print("desired_input_types", desired_input_types) # the desired input types for the first dsl
        # print("desired_output_type", desired_output_type) # the desired output type for the second dsl

        dsl_2_pool = self.general_type_to_primitive_function_mapping[desired_output_type]
        print("dsl_2_pool:", dsl_2_pool)
        # dsl_2_func = sum(dsl_2_pool.values(), []) # all possible dsl_2 candidates
        dsl_2_func = [item for sublist in dsl_2_pool.values() for item in sublist]  #### ****
        

        # print("dsl_2_func:", dsl_2_func)


        all_keys_flat = sum(dsl_2_pool.keys(), ())
        dsl_2_input_pool = list(set(all_keys_flat)) # all possible input types for dsl_2 candidates
        if 'ContainerContainer' in dsl_2_input_pool:
            dsl_2_input_pool.remove('ContainerContainer')

        # print("dsl_2_input_pool:", dsl_2_input_pool)

        dsl_1_pool = [self.general_type_to_primitive_function_mapping[key] for key in dsl_2_input_pool]
        # print("dsl_1_pool:", dsl_1_pool)

        # dsl_1_pool_filtered = {}
        # # filtering by the desired input and the available types for dsl_1

        # for d in dsl_1_pool:
        #     # Filter keys that contain all elements in desired_input_types and only have elements in available_types
        #     for key, value in d.items():
        #         if all(element in key for element in desired_input_types):
        #             if all(element in available_types for element in key):
        #                 # Merge the filtered results
        #                 if key in dsl_1_pool_filtered:
        #                     dsl_1_pool_filtered[key].extend(value)
        #                 else:
        #                     dsl_1_pool_filtered[key] = value

        dsl_1_pool_filtered = {key: value for d in dsl_1_pool  # optimized
                        for key, value in d.items() 
                        if all(element in key for element in desired_input_types) and 
                           all(element in available_types for element in key)}



        # print("dsl_1_pool_filtered:", dsl_1_pool_filtered)
        dsl_1_func = sum(dsl_1_pool_filtered.values(), []) # all possible dsl_1 candidates
        dsl_1_func = list(set(dsl_1_func)) # get only unique dsl_1 candidates
        # print("dsl_1_func:", dsl_1_func)

        possible_pairs = []
        weights_of_pairs = []


        # for dsl_1 in dsl_1_func:
        #     for dsl_2 in dsl_2_func:
        #         dsl_1_output_type = self.primitive_function_to_general_type_mapping[dsl_1]['output']
        #         dsl_2_input_types = self.primitive_function_to_general_type_mapping[dsl_2]['inputs']
        #         if dsl_1_output_type in dsl_2_input_types: # the output type of dsl1 must be in the input types of dsl2
        #             if len(dsl_2_input_types) == 1: # if the only input type of dsl2 == output type of dsl1

        #                 possible_pairs.append((dsl_1, dsl_2))
        #                 weights_of_pairs.append(dependence_weights.loc[dsl_1, dsl_2]+0.1)
        #             else: # if dsl2 has more input types
        #                 other_input_types = [type_ for type_ in dsl_2_input_types if type_ != dsl_1_output_type]
        #                 if all(type_ in available_types for type_ in other_input_types): # the rest of input types need to be in available types
        #                     possible_pairs.append((dsl_1, dsl_2))
        #                     weights_of_pairs.append(dependence_weights.loc[dsl_1, dsl_2]+0.1)
        

        for dsl_1 in dsl_1_func:
            dsl_1_output_type = self.primitive_function_to_general_type_mapping[dsl_1]['output']
            for dsl_2 in dsl_2_func:
                dsl_2_input_types = self.primitive_function_to_general_type_mapping[dsl_2]['inputs']
                if dsl_1_output_type in dsl_2_input_types:
                    if len(dsl_2_input_types) == 1 or all(t in available_types for t in dsl_2_input_types if t != dsl_1_output_type):
                        possible_pairs.append((dsl_1, dsl_2))
                        weights_of_pairs.append(dependence_weights.loc[dsl_1, dsl_2] + 0.1) ### optimized




        # print("possible_pairs", possible_pairs)
        # print("weights_of_pairs", weights_of_pairs)

        print(len(possible_pairs), len(weights_of_pairs))

        selected_pair = random.choices(possible_pairs, weights= weights_of_pairs, k=1)[0]
        # print("selected pair:", selected_pair)

        return selected_pair[0], selected_pair[1]






    def amplify_into_two(self, node_to_amplify, assignments): # *****


        
        func_of_node = node_to_amplify.value.func.id

        args_of_node = [node_to_amplify.value.args[i].id for i in range(len(node_to_amplify.value.args))]

        print("func:",func_of_node)
        print("args:", args_of_node)
        desired_input_types = []
        desired_inputs_types_mapping = {} # the inputs that will be necessary after amplifying
        desired_output_type = self.primitive_function_to_general_type_mapping[func_of_node]["output"]
        for i in range(len(node_to_amplify.value.args)):
            if node_to_amplify.value.args[i].id not in self.primitive_constant_to_type_mapping.keys():
                desired_input = node_to_amplify.value.args[i].id
                desired_input_type = self.type_inferer.type_dict[desired_input][0]
                desired_input_types.append(desired_input_type)
                if desired_input_type in desired_inputs_types_mapping:
                    desired_inputs_types_mapping[desired_input_type] += (desired_input,)
                else:
                    desired_inputs_types_mapping[desired_input_type]= (desired_input,)
        # print('target',node_to_amplify.targets[0].id)
        # print(len(assignments))
        if node_to_amplify.targets[0].id.startswith('x'):
            node_target_no = int(node_to_amplify.targets[0].id[1:])
        else:
            node_target_no = len(assignments)
        # print('node target:', node_target_no)
        existing_types = ['Grid']
        existing_types_to_targets = {'Grid':['I']}
        if node_target_no > 1:
            for i in range(node_target_no-1):
                new_type = self.type_inferer.type_dict['x'+str(i+1)][0]
                existing_types.append(new_type)
                if new_type in existing_types_to_targets:
                    existing_types_to_targets[new_type].append('x'+str(i+1))
                else:
                    existing_types_to_targets[new_type] = ['x'+str(i+1)]

        existing_types = list(set(existing_types)) # this is the types that are available to use at the node to amplify
        # print("existing types:", existing_types)
        # print('existing_types_to_targets', existing_types_to_targets)


        



        # print('----------------type_dict-----------------')
        # print(self.type_inferer.type_dict)
        # print('------------primitive_constant_to_type_mapping--------------')
        # print(self.primitive_constant_to_type_mapping)
        # print('-----------------CONSTANT_TO_TYPE_MAPPING-------------------')
        # print(CONSTANT_TO_TYPE_MAPPING)
        # print('-----------------primitive_function_to_general_type_mapping--------------------')
        # print(self.primitive_function_to_general_type_mapping)
        # print('-------------------general_type_to_primitive_function_mapping-------------------------')
        # print(self.general_type_to_primitive_function_mapping)
        # print(self.general_type_to_primitive_function_mapping["Grid"])
        # print('------------type_to_primitive_constant_mapping--------------')
        # print(self.type_to_primitive_constant_mapping)

        dsl1, dsl2 = self.sample_two_func_amplified(desired_input_types,desired_output_type, existing_types)

        # print("desired input:", desired_input_types)
        # print("desired output:", desired_output_type)
        print("dsl 1:", dsl1)
        print("dsl 2:", dsl2)
        dsl_1_input_types = self.primitive_function_to_general_type_mapping[dsl1]['inputs']
        dsl_1_output_type = self.primitive_function_to_general_type_mapping[dsl1]['output']
        dsl_2_input_types = self.primitive_function_to_general_type_mapping[dsl2]['inputs']
        # dsl_2_output_type = self.primitive_function_to_general_type_mapping[dsl2]['output']

        ### next will be try to update these two funcs in the program ast, also bump up the order of each x_n

        print("program_ast:", ast.unparse(self.program_ast))
        
        # start idx = self.memory_index - 1
        self.bump_up_variable_names(self.memory_index)

        # new_line_1 = ast.Assign(
        #     targets=[ast.Name(id='x0', ctx=ast.Store())],
        #     value=ast.Call(func=ast.Name(id='righthalf', ctx=ast.Load()), args=[ast.Name(id='I', ctx=ast.Load())], keywords=[]),
        #     lineno=0#,  # Setting lineno
        #     # col_offset=4  # Setting col_offset
        # )

        # self.program_ast.body[0].body.insert(0, new_line_1)
        new_target_1 = 'x'+str(node_target_no)
        # new_target_2 = 'x'+str(node_target_no+1)

        

        new_args = []
        # print('desired inputs', desired_inputs_types_mapping

        ### get args for the first line
        for input_type in dsl_1_input_types:
            if len(sum(desired_inputs_types_mapping.values(), ()))>0: # if there's still desired input to be designated
                if input_type not in desired_inputs_types_mapping: # if the type is not our desired type
                    if input_type in existing_types_to_targets:
                        new_args.append(random.choice(existing_types_to_targets[input_type]))
                    else:
                        new_args.append(random.choice(self.type_to_primitive_constant_mapping[input_type]))

                elif desired_inputs_types_mapping[input_type] == (): # if target of this type is empty
                    if input_type in existing_types_to_targets:
                        new_args.append(random.choice(existing_types_to_targets[input_type]))
                    else:
                        new_args.append(random.choice(self.type_to_primitive_constant_mapping[input_type]))
                else: # if the targets of this type is not empty
                    arg = random.choice(desired_inputs_types_mapping[input_type]) # randomly select one from the available targets, normally there's only 1
                    new_args.append(arg)
                    temp_list = list(desired_inputs_types_mapping[input_type])
                    temp_list.remove(arg)
                    desired_inputs_types_mapping[input_type] = tuple(temp_list)
            else:  # if all desired inputs are designated already
                if input_type in existing_types_to_targets:
                    new_args.append(random.choice(existing_types_to_targets[input_type]))
                else:
                    new_args.append(random.choice(self.type_to_primitive_constant_mapping[input_type]))


        new_line_1 = ast.parse(f"{new_target_1} = {dsl1}({','.join(new_args)})")
        self.program_ast.body[0].body.insert(node_target_no-1, new_line_1)

        desired_inputs_types_mapping = {dsl_1_output_type: (new_target_1,)}
        if dsl_1_output_type in existing_types_to_targets:
            existing_types_to_targets[dsl_1_output_type].append(new_target_1)
        else:
            existing_types_to_targets[dsl_1_output_type] = [new_target_1]
        

        new_args = []

        ### get args for the second line
        for input_type in dsl_2_input_types:
            if len(sum(desired_inputs_types_mapping.values(), ()))>0: # if there's still desired input to be designated
                if input_type not in desired_inputs_types_mapping: # if the type is not our desired type
                    if input_type in existing_types_to_targets:
                        new_args.append(random.choice(existing_types_to_targets[input_type]))
                    else:
                        new_args.append(random.choice(self.type_to_primitive_constant_mapping[input_type]))
                # elif desired_inputs_types_mapping[input_type] == (): # if target of this type is empty
                #     if input_type in existing_types_to_targets:
                #         new_args.append(random.choice(existing_types_to_targets[input_type]))
                #     else:
                #         new_args.append(random.choice(self.type_to_primitive_constant_mapping[input_type]))
                else: # if the targets of this type is not empty
                    arg = random.choice(desired_inputs_types_mapping[input_type]) # randomly select one from the available targets, normally there's only 1
                    new_args.append(arg)
                    desired_inputs_types_mapping[input_type] = ()
            else:  # if all desired inputs are designated already
                if input_type in existing_types_to_targets:
                    new_args.append(random.choice(existing_types_to_targets[input_type]))
                else:
                    new_args.append(random.choice(self.type_to_primitive_constant_mapping[input_type]))

        new_node = ast.parse(f"{dsl2}({','.join(new_args)})").body[0].value
        assignments[node_target_no-1].value = new_node


        print("program_ast:", ast.unparse(self.program_ast))

        return dsl1, dsl2 ## this part might needs optimization to make it run faster
    


    def mutate3(self):
        method_to_go = random.choices(["mutate","replace by two"], weights=[0, 1])[0]
        mutation = []

        if method_to_go == "mutate":
        
            # mutation_node_weights = [0.27067, 0.27067, 0.18045, 0.09022, 0.03609] # Poisson lambda == 2
            mutation_node_weights = [0.36788, 0.18394, 0.06131] # Poisson lambda = 1

            assignments = [node for node in ast.walk(self.program_ast) if isinstance(node, ast.Assign)]
            if len(assignments) < 4:
                n_nodes = 1
            else:
                n_nodes = 3

            if n_nodes > 1:
                n_weights = [w / sum(mutation_node_weights[:n_nodes]) for w in mutation_node_weights[:n_nodes]]
                n_mutation = random.choices(range(1,n_nodes+1), weights = n_weights)[0]
            else:
                n_mutation = 1
            # print("---------------n mutation:",n_mutation)
            

            nodes_to_mutate = random.sample(assignments, k = n_mutation)
            for node_to_mutate in nodes_to_mutate:
                mutation_choice = random.choices(
                    ["replace_argument", "replace_function", "replace_variable"],
                    weights=[self.phi_arg, self.phi_func, 1 - (self.phi_func + self.phi_arg)],
                )[0]
                self.memory_index = assignments.index(node_to_mutate) + 1
                if mutation_choice == "replace_argument":
                    arg_to_replace_id = random.choice(range(len(node_to_mutate.value.args)))
                    arg_to_replace = node_to_mutate.value.args[arg_to_replace_id].id
                    # print("arg_to_replace:", arg_to_replace)
                    new_arg, new_variable_mutation = self.replace_argument(arg_to_replace=arg_to_replace)
                    assignments[self.memory_index - 1].value.args[arg_to_replace_id].id = new_arg
                    if new_variable_mutation:
                        mutation.append(("arg", arg_to_replace, new_arg, new_variable_mutation)) ##
                    else:
                        mutation.append(("arg", arg_to_replace, new_arg)) ##
                elif mutation_choice == "replace_function":
                    function_to_replace = node_to_mutate.value.func.id
                    # print("function_to_replace:", function_to_replace)
                    new_function = self.replace_function(function_to_replace=function_to_replace)
                    assignments[self.memory_index - 1].value.func.id = new_function
                    mutation.append(("func", function_to_replace, new_function)) ##
                else:
                    variable_to_replace = node_to_mutate.targets[0].id
                    # print("variable_to_replace:", variable_to_replace)
                    new_variable_value = self.replace_variable(variable_to_replace=variable_to_replace)
                    mutation.append((
                        "var_def",
                        f"{node_to_mutate.value.func.id}({', '.join([arg.id for arg in node_to_mutate.value.args])})",
                        new_variable_value,
                    ))
                    new_node = ast.parse(new_variable_value).body[0].value
                    assignments[self.memory_index - 1].value = new_node
        else:

            assignments = [node for node in ast.walk(self.program_ast) if isinstance(node, ast.Assign)]
            node_to_amplify = random.choice(assignments)
            self.memory_index = assignments.index(node_to_amplify) + 1
            function_to_amplify = node_to_amplify.value.func.id
            if function_to_amplify.startswith('x'): # if starts with x, not amplify but replace func
                new_function = self.replace_function(function_to_replace=function_to_amplify)
                assignments[self.memory_index - 1].value.func.id = new_function
                mutation.append(("func", function_to_amplify, new_function))
            else:
                amplified_func_1, amplified_func_2 = self.amplify_into_two(node_to_amplify, assignments)
                mutation.append(("func_amplified", function_to_amplify, amplified_func_1, amplified_func_2))




        return mutation







    def sample_term_with_type(self, term_type, terms_to_exclude):
        filtered_type_dict = self.filter_type_dict_by_index()
        candidate_terms = []
        # add primitive functions
        if isinstance(term_type, dict):
            candidate_terms += self.general_type_to_primitive_function_mapping[term_type["output"]][
                term_type["inputs"]
            ]
        elif isinstance(term_type, Arrow):
            candidate_terms += self.base_type_to_primitive_function_mapping[display_type(Arrow)]
        # add primitive constants
        else:
            if term_type in self.type_to_primitive_constant_mapping.keys():
                candidate_terms += self.type_to_primitive_constant_mapping[term_type]
        # add variables in memory
        for var_name, var_type in filtered_type_dict.items():
            if var_type[0] == term_type:
                candidate_terms.append(var_name)
        # filter out excluded terms
        candidate_terms = [term for term in candidate_terms if term not in terms_to_exclude]
        if not candidate_terms:
            raise ValueError(f"No candidate terms of type {display_type(term_type)} found")
        return random.choice(candidate_terms)

    def sample_function_with_output_type(self, output_type):
        filtered_type_dict = self.filter_type_dict_by_index()
        candidate_terms = []
        # add primitive functions
        for input_type in self.general_type_to_primitive_function_mapping[output_type]:
            candidate_terms += self.general_type_to_primitive_function_mapping[output_type][
                input_type
            ]
        # add variables in memory
        for var_name, var_type in filtered_type_dict.items():
            if var_type[0].startswith("Arrow") and var_type[0].endswith(f", {output_type})"):
                candidate_terms.append(var_name)
        # filter out functions that are do not have base type hints
        candidate_terms = [
            candidate_term
            for candidate_term in candidate_terms
            if candidate_term in self.primitive_function_to_base_type_mapping.keys()
        ]
        if not candidate_terms:
            raise ValueError(
                f"No candidate functions with output type {display_type(output_type)} found"
            )
        return random.choice(candidate_terms)

    def filter_memory_by_index(self):
        filtered_memory = {"I": self.type_inferer.memory["I"]}
        for var_index in range(1, self.memory_index):
            var_name = f"x{var_index}"
            filtered_memory[var_name] = self.type_inferer.memory[var_name]
        return filtered_memory

    def filter_type_dict_by_index(self):
        filtered_type_dict = {"I": self.type_inferer.type_dict["I"]}
        for var_index in range(1, self.memory_index):
            var_name = f"x{var_index}"
            filtered_type_dict[var_name] = self.type_inferer.type_dict[var_name]
        return filtered_type_dict

    def bump_up_variable_names(self, start_idx):
        # bump up variable names in program ast
        for node in ast.walk(self.program_ast):
            if isinstance(node, ast.Name) and node.id.startswith("x"):
                var_num = int(node.id[1:])
                if var_num >= start_idx:
                    node.id = f"x{var_num + 1}"
        # bump up variable names in type inferer and memory
        for var_name in copy.copy(list(self.type_inferer.type_dict.keys())):
            if var_name.startswith("x") and int(var_name[1:]) >= start_idx:
                self.type_inferer.type_dict[
                    f"x{int(var_name[1:])+1}r"
                ] = self.type_inferer.type_dict.pop(var_name)
                self.type_inferer.memory[
                    f"x{int(var_name[1:]) + 1}r"
                ] = self.type_inferer.memory.pop(var_name)
        for var_name in copy.copy(list(self.type_inferer.type_dict.keys())):
            if var_name.startswith("x") and var_name.endswith("r"):
                self.type_inferer.type_dict[var_name[:-1]] = self.type_inferer.type_dict.pop(
                    var_name
                )
                self.type_inferer.memory[var_name[:-1]] = self.type_inferer.memory.pop(var_name)

    def add_variable_to_ast(self, index_to_insert, new_variable_value):
        self.bump_up_variable_names(index_to_insert + 1)
        new_assignment = ast.parse(f"x{index_to_insert+1} = {new_variable_value}").body[0]
        for node in ast.walk(self.program_ast):
            if isinstance(node, ast.FunctionDef):
                node.body.insert(index_to_insert, new_assignment)
                break

    def add_variable_to_memory(self, index_to_insert, new_variable_value):
        local_memory = self.filter_memory_by_index()
        for primitive in list(self.primitive_function_to_general_type_mapping.keys()) + list(
            self.primitive_constant_to_type_mapping.keys()
        ):
            local_memory[primitive] = eval(primitive)
        new_variable = eval(new_variable_value, {}, local_memory)
        self.type_inferer.memory[f"x{index_to_insert}"] = new_variable

    def add_variable_type_dict(self, index_to_insert, new_variable_type):
        self.type_inferer.type_dict[f"x{index_to_insert}"] = [new_variable_type]

    def replace_argument(self, arg_to_replace):
        """replace an argument with an argument of the same type, add new variable with probability phi_var and replace argument with this new variable"""
        if arg_to_replace.startswith("x") or arg_to_replace == "I":
            arg_type = self.type_inferer.type_dict[arg_to_replace][0]
            if isinstance(arg_type, Arrow):
                raise ValueError(
                    f"argument type {display_type(arg_type)} replacement not supported"
                )
        elif arg_to_replace in self.primitive_constant_to_type_mapping.keys():
            arg_type = CONSTANT_TO_TYPE_MAPPING[arg_to_replace]
        elif arg_to_replace in self.primitive_function_to_general_type_mapping.keys():
            arg_type = self.primitive_function_to_general_type_mapping[arg_to_replace]
        else:
            raise ValueError(f"argument {arg_to_replace} not found")
        if random.random() < self.phi_var:
            new_variable_type = arg_type
            new_variable_value = self.create_variable(new_variable_type)
            self.add_variable_to_ast(
                index_to_insert=self.memory_index - 1, new_variable_value=new_variable_value
            )
            self.add_variable_to_memory(
                index_to_insert=self.memory_index - 1, new_variable_value=new_variable_value
            )
            self.add_variable_type_dict(
                index_to_insert=self.memory_index - 1, new_variable_type=new_variable_type
            )
            new_arg = f"x{copy.copy(self.memory_index)}"
            # self.memory_index += 1
            mutation = new_variable_value
        else:
            new_arg = self.sample_term_with_type(
                term_type=arg_type, terms_to_exclude=[arg_to_replace]
            )
            mutation = None
        return new_arg, mutation

    def replace_function(self, function_to_replace):
        """replace a function with a function of the same type"""
        if function_to_replace.startswith("x"):
            function_type = self.type_inferer.type_dict[function_to_replace]
            if "Callable" in display_type(function_type):
                raise ValueError(
                    f"function type {display_type(function_type)} replacement not supported"
                )
            else:
                new_function = self.sample_term_with_type(
                    term_type=function_type, terms_to_exclude=[function_to_replace]
                )
        else:
            function_type = self.primitive_function_to_general_type_mapping[function_to_replace]
            new_function = self.sample_term_with_type(
                term_type=function_type, terms_to_exclude=[function_to_replace]
            )
        return new_function

    def create_variable(self, variable_type):
        args = []
        if isinstance(variable_type, Arrow):
            raise ValueError(
                f"variable type {display_type(variable_type)} replacement not supported"
            )
        else:
            new_func = self.sample_function_with_output_type(output_type=variable_type)
            if new_func.startswith("x"):
                new_func_type = self.type_inferer.type_dict[new_func][0]
            else:
                new_func_type = random.choice(
                    self.primitive_function_to_base_type_mapping[new_func]
                )
            for arg_type in new_func_type.inputs:
                arg = self.sample_term_with_type(term_type=arg_type, terms_to_exclude=[])
                args.append(arg)
        new_variable_value = f"{new_func}({','.join(args)})"
        return new_variable_value

    def replace_variable(self, variable_to_replace):
        """replace a variable with a variable of the same type"""
        variable_type = self.type_inferer.type_dict[variable_to_replace][0]
        new_variable_value = self.create_variable(variable_type=variable_type)
        return new_variable_value
