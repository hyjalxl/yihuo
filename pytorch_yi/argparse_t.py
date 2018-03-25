# coding=utf-8
# user=hu

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('echo', help='echo the string you use here')
parser.add_argument('echo_2', help='echo the string you use here', type=int)

args = parser.parse_args()
print(args.echo)
print(args.echo_2 * 2)

if __name__ == "__main__":
    pass