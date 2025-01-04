import pickle

# Define the dummy data
hlvis_annotation = {
    'P0003video2_00-01-18_0001959': {
        'num_instance_masks': 3,
        'hierarchy': {
            1: {'parent': None, 'children': []},
            2: {'parent': None, 'children': []},
            3: {'parent': None, 'children': []}
        }
    }
}

# Specify the file name for the pickle file
output_file = 'hannotation.pickle'

# Write the data to a pickle file
with open(output_file, 'wb') as f:
    pickle.dump(hlvis_annotation, f)

print(f"Dummy pickle file '{output_file}' has been created.")
