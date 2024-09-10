import os
import subprocess
import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Python version of the bash script")
    parser.add_argument('remark', type=str, help='First argument: remark')
    parser.add_argument('d', nargs='+', help='Third argument: dataset names (multiple values allowed)', choices=list(['a','b','c'])+['all'])

    return parser.parse_args()


args=parse_args()
print(args)