import ast
from collections import defaultdict

# Helper function to extract function calls and their input/output arguments from an AST node
def extract_function_calls(node):
    calls = []
    input_functions = set()
    output_functions = set()

    for child in ast.walk(node):
        if isinstance(child, ast.Assign):  # Handling assignments (e.g., x = func())
            if isinstance(child.value, ast.Call):
                if isinstance(child.value.func, ast.Name):  # Function call
                    func_name = child.value.func.id
                    calls.append(func_name)

                    # Check if "I" is in the arguments
                    for arg in child.value.args:
                        if isinstance(arg, ast.Name) and arg.id == "I":
                            input_functions.add(func_name)
                    
                    # Check if the assigned variable is "O"
                    for target in child.targets:
                        if isinstance(target, ast.Name) and target.id == "O":
                            output_functions.add(func_name)

    return calls, input_functions, output_functions

# Analyze a single program and extract function call sequences, inputs, and outputs
def analyze_program(program_code):
    # Parse the program code into an AST
    tree = ast.parse(program_code)
    
    # Extract function call sequences along with input/output tracking
    function_calls, input_functions, output_functions = extract_function_calls(tree)
    
    # Build connectivity information
    connectivity = defaultdict(list)
    for i in range(len(function_calls) - 1):
        current_func = function_calls[i]
        next_func = function_calls[i + 1]
        connectivity[current_func].append(next_func)
    
    return connectivity, input_functions, output_functions

# Example program as a string
program_code = """
def solve_7468f01a(I):
    x1 = objects(I, F, T, T)
    x2 = first(x1)
    x3 = subgrid(x2, I)
    O = vmirror(x3)
    return O
"""

# Analyze the example program
connectivity, input_functions, output_functions = analyze_program(program_code)

# Display the connectivity information
print("Connectivity information:")
for func, follows in connectivity.items():
    print(f"Function '{func}' is followed by: {follows}")

# Display which functions take "I" as input
print("\nFunctions that take 'I' as an input argument:")
for func in input_functions:
    print(f"Function '{func}'")

# Display which function outputs to "O"
print("\nFunctions that output to 'O':")
for func in output_functions:
    print(f"Function '{func}'")

# Example output storage: Convert connectivity to a node-based structure
nodes = {}
for func, follows in connectivity.items():
    nodes[func] = {
        'follows': follows,
        'takes_I_as_input': func in input_functions,
        'outputs_to_O': func in output_functions
    }

print("\nNode structure:")
for func, info in nodes.items():
    print(f"{func}: {info}")
