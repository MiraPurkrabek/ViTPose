import os
import shutil


def main(workdir="work_dirs", tf_logs="tensorboard_logs"):
    os.makedirs(tf_logs, exist_ok=True)
    
    for subdir in os.listdir(workdir):
        subdir_path = os.path.join(workdir, subdir)

        if not os.path.isdir(subdir_path):
            continue

        tf_logs_dir = os.path.join(subdir_path, "tf_logs")

        if not os.path.exists(tf_logs_dir) or not os.path.isdir(tf_logs_dir):
            continue

        counter = 0
        for tf_log in os.listdir(tf_logs_dir):
            shutil.copy(
                os.path.join(tf_logs_dir, tf_log),
                os.path.join(tf_logs, tf_log),
                # os.path.join(tf_logs, "{:s}_{:02d}".format(subdir, counter)),
            )
            counter += 1
        

if __name__ == "__main__":
    main()