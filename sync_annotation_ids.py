import json
import argparse
import numpy as np

def parse_args():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Sync annotation ids')
    parser.add_argument('source_file', help='Annotation file')
    parser.add_argument('target_file', help='Annotation file')

    args = parser.parse_args()
    return args

def main(args):

    src = json.load(open(args.source_file))
    tgt = json.load(open(args.target_file))

    if isinstance(tgt, list) or "annotations" not in tgt.keys():
        tgt = {
            "info": src["info"],
            "licenses": src["licenses"],
            "images": [],
            "annotations": tgt
        }
    elif "images" not in tgt.keys():
        tgt["images"] = []
    elif "licenses" not in tgt.keys():
        tgt["licenses"] = src["licenses"]
    elif "categories" not in tgt.keys():
        tgt["categories"] = src["categories"]

    print(src.keys())
    print(tgt.keys())

    for img in src["images"]:
        for img_tgt in tgt["images"]:
            if img["file_name"] == img_tgt["file_name"]:
                
                for ann in tgt["annotations"]:
                    if ann["image_id"] == img_tgt["id"]:
                        ann["image_id"] = img["id"]

    # If annotations has to ID, add a random one
    for ann in tgt["annotations"]:
        if "id" not in ann.keys():
            ann["id"] = np.random.randint(0, 100000000)

        if "bbox" not in ann.keys():
            ann["bbox"] = [0, 0, 0, 0]

    tgt["images"] = src["images"]                    

    json.dump(tgt, open(args.target_file, "w"), indent=2)

if __name__ == "__main__":
    args = parse_args()
    main(args)
            