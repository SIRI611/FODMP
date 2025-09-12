import os
import numpy as np

DATA_PATH = "datasets/soccer_success_dataset_0.npz"

def test_dataset(path=DATA_PATH):
    """
    Preview a saved dataset by loading it from file and printing
    out some information about the episodes it contains.

    Args:
        path (str): Path to saved dataset file (default: DATA_PATH)
    """
    if not os.path.exists(path):
        print(f"[!] Dataset file not found: {path}")
        return

    data = np.load(path, allow_pickle=True)
    episodes = data["episodes"].tolist()  # convert from np.object_ to list of dicts

    print(f"\nLoaded {len(episodes)} successful episodes from {path}\n")

    # Preview first few episodes
    for i, ep in enumerate(episodes[:3]):
        obs_shape = ep["observations"].shape
        act_shape = ep["actions"].shape
        rew_sum   = float(np.sum(ep["rewards"]))
        length    = len(ep["dones"])
        print(f"Episode {i}: len={length}, obs={obs_shape}, act={act_shape}, total_reward={rew_sum:.2f}")

    # Inspect one episode in detail
    if episodes:
        ep0 = episodes[0]
        print("\nExample Episode[0] keys:", list(ep0.keys()))
        print(" First obs:", ep0["observations"][0])
        print(" First act:", ep0["actions"][0])
        print(" First rew:", ep0["rewards"][0], "done?", ep0["dones"][0])

if __name__ == "__main__":
    test_dataset()