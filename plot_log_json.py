import os
import argparse
import numpy as np
import json
import matplotlib.pyplot as plt


def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('filepath', help='')
    parser.add_argument('--mode', default='train')
    parser.add_argument('--value', default='')
    parser.add_argument('--window_size', type=int, default=50)

    args = parser.parse_args()
    if args.value == '':
        args.value = 'loss' if args.mode == 'train' else 'AP'
    
    return args

def main(args):
    
    values = []
    horizontal = []

    filename = os.path.basename(args.filepath)
    save_path = os.path.join(
        os.path.dirname(args.filepath),
        "plot_{:s}_{:s}_{:s}.png".format(args.mode, args.value, filename)
    )

    vertical_label = args.value

    horiz_counter = 0
    with open(args.filepath, "r") as fp:  
        for line in fp.readlines():
            line_dct = json.loads(line)
            if not 'mode' in line_dct.keys():
                continue
            mode = line_dct['mode']
            
            if mode != args.mode:
                continue

            if args.mode == "val":
                value = float(line_dct[args.value])
                horiz = int(line_dct["epoch"])
            elif args.mode == "train":
                value = float(line_dct[args.value])
                horiz = horiz_counter
                horiz_counter += 1
                # horiz = (int(line_dct["epoch"])-1)*1000 + int(line_dct["iter"])
                
            values.append(value)
            horizontal.append(horiz)

    values = np.array(values)
    horizontal = np.array(horizontal)

    # if args.mode == "train":
    #     horizontal = horizontal / 1000

    plt.plot(horizontal, values)
    n = args.window_size
    plt.plot(horizontal[n-1:], moving_average(values, n=n))
    plt.grid(True)
    plt.ylabel(vertical_label)
    # plt.ylim([-0.01, 0.01])
    plt.xlabel("Epoch")

    # plt.show()
    plt.savefig(save_path)


if __name__ == '__main__':
    args = parse_args()
    main(args)