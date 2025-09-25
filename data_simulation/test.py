import h5py
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str, required=True)
args = parser.parse_args()

# Open file in read mode
with h5py.File(args.file, "r") as f:
    # List all groups
    # print("Keys:", list(f.keys()))
    def print_structure(name, obj):
        print(name, "->", type(obj))
    # f.visititems(print_structure)
    # Access a dataset
    dataset = f["traj_1"]  # Read the first trajectory
    dataset.visititems(print_structure)
    # print(dataset["env_states"]["actors"]["table"][:])
    print(dataset["actions"][:])