"""
Extract dsl functions' input output data types
"""
import json



def extract_dsl_functions(file_path, input_dict, output_dict):
    with open(file_path, 'r') as file:
        content = file.read()

    functions = content.split('def ')[1:]

    dsl_functions = []

    for func in functions:
        # Find the function name
        func_name_end = func.find('(')
        func_name = func[:func_name_end].strip()

        # Extract parameters
        params_start = func.find('(') + 1
        params_end = func.find(')')
        params_str = func[params_start:params_end].strip()

        # Handle empty parameters
        input_types = []
        if params_str:
            params = params_str.split(',')
            for param in params:
                if ':' in param:
                    param_type = param.split(':')[1].strip()
                else:
                    param_type = "Any"  # Default if no type hint is provided
                input_types.append(param_type)

        # Extract the return type
        arrow_index = func.find('->')

        colon_index = func.find(':', arrow_index)
        output_type = func[arrow_index + 2:colon_index].strip()


        for input_type in input_types:
            if input_type in list(input_dict.keys()):
                input_dict[input_type].append(func_name)
            else:
                input_dict[input_type] = [func_name]
        
        if output_type in list(output_dict.keys()):
            output_dict[output_type].append(func_name)
        else:
            output_dict[output_type] = [func_name]


        # dsl_functions.append((func_name, input_types, output_type))

    # return dsl_functions
    return input_dict, output_dict


def main():
    file_path = "codeit/dsl/dsl.py"
    input_type_func = {}
    output_type_func = {}
    input_type_func, output_type_func = extract_dsl_functions(file_path, input_type_func, output_type_func)
    
    with open("mutate_weights/input_type_func.json", "w") as json_file:
        json.dump(input_type_func, json_file, indent=4)
    
    with open("mutate_weights/output_type_func.json", "w") as json_file:
        json.dump(output_type_func, json_file, indent=4)


    with open("mutate_weights/input_type_func.json", "r") as json_file:
        input_type_func = json.load(json_file)

    with open("mutate_weights/output_type_func.json", "r") as json_file:
        output_type_func = json.load(json_file)

    # for func_name, input_types, output_type in dsl_functions:
    #     print(f"Function: {func_name}")
    #     print(f"Input Types: {input_types}")
    #     print(f"Output Type: {output_type}")
    #     print()
    print('------------input-------------')
    for key in input_type_func.keys():
        print("type:", key)
        print(input_type_func[key])
    print('------------output-------------')
    for key in output_type_func.keys():
        print("type:", key)
        print(output_type_func[key])

    

if __name__ == "__main__":
    main()
