import os
import shutil
import argparse
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Prune work dirs")
    parser.add_argument(
        "--workdir", default="work_dirs", help="workdir directory to prune"
    )
    parser.add_argument("--num-keep", default=3, type=int, help="Number of checkpoints to keep")
    parser.add_argument("--keep-method", default="uniform", type=str, help="How to select checkpoints to keep. Either 'uniform' or 'last'")
    parser.add_argument("--dry", action="store_true", help="Dry run. Do not delete anything just print it")
    
    args = parser.parse_args()

    assert args.num_keep >= 1, "num_keep must be >= 1"
    assert args.keep_method.lower() in ["uniform", "last"], "keep_method must be either 'uniform' or 'last'"

    return args


def load_checkpoint_numbers(folder_path):
    checkpoints = []
    for checkpoint_name in os.listdir(folder_path):
        if not checkpoint_name.startswith("epoch_"):
            continue

        # Get rid of the extension
        checkpoint_name = checkpoint_name.split(".")[0]

        # Get rid of the epoch_ prefix
        checkpoint_str = checkpoint_name.split("_")[1]
        checkpoint_number = int(checkpoint_str)
        checkpoints.append(checkpoint_number)

    return checkpoints


def select_checkpoints(checkpoints, num_checkpoints, method="uniform"):
    nums = np.array(checkpoints)
    nums = np.sort(nums)
    
    selected_checkpoints = []
    num_checkpoints = min(num_checkpoints, len(nums))

    if method.lower() == "uniform":
        # Make sure the last checkpoint is always selected
        selected_checkpoints.append(nums[-1])

        # Select the rest of the checkpoints uniformly
        num_checkpoints -= 1
        num_checkpoints = min(num_checkpoints, len(nums) - 1)
        if num_checkpoints > 0:
            step_size = max(len(nums) // num_checkpoints, 1)

            selected_checkpoints += list(nums[::step_size])

    elif method.lower() == "last":
        # Select the last checkpoints
        selected_checkpoints = nums[-num_checkpoints:]

    else:
        raise ValueError("Unknown method: {:s}".format(method))

    return selected_checkpoints


def main(args):
    deleted_mb_all = 0

    for subdir in os.listdir(args.workdir):
        subdir_path = os.path.join(args.workdir, subdir)

        if not os.path.isdir(subdir_path):
            continue

        checkpoints = load_checkpoint_numbers(subdir_path)
        if len(checkpoints) == 0:
            continue
        
        selected_checkpoints = select_checkpoints(checkpoints, args.num_keep, args.keep_method)

        print("Pruning {:s}...\n--> there is {:d} checkpoints, deleting {:d} of them".format(
            subdir_path,
            len(checkpoints),
            len(checkpoints) - len(selected_checkpoints),
        ))

        deleted_size = 0
        for checkpoint_name in os.listdir(subdir_path):
            if not checkpoint_name.startswith("epoch_"):
                continue

            # Get rid of the extension
            checkpoint_path = os.path.join(subdir_path, checkpoint_name)
            checkpoint_name = checkpoint_name.split(".")[0]

            # Get rid of the epoch_ prefix
            checkpoint_str = checkpoint_name.split("_")[1]
            checkpoint_number = int(checkpoint_str)

            if checkpoint_number not in selected_checkpoints:
                deleted_size += os.path.getsize(checkpoint_path)
                if args.dry:
                    print("\tWould delete: {:s}".format(checkpoint_path))
                else:
                    os.remove(checkpoint_path)
        
        deleted_size_mb = deleted_size / 1024 / 1024
        deleted_size_gb = deleted_size_mb / 1024
        deleted_mb_all += deleted_size_mb
        if args.dry:
            print("\tWould delete {:.2f} MB ({:.2f} GB)".format(deleted_size_mb, deleted_size_gb))
        else:
            print("\tDeleted {:.2f} MB ({:.2f} GB)".format(deleted_size_mb, deleted_size_gb))
            
    if args.dry:
        print("Would delete {:.2f} MB ({:.2f} GB) in total".format(deleted_mb_all, deleted_mb_all / 1024))
    else:
        print("Deleted {:.2f} MB ({:.2f} GB) in total".format(deleted_mb_all, deleted_mb_all / 1024))

if __name__ == "__main__":
    args = parse_args()
    main(args)