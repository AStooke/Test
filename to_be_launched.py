
import sys
import argparse
import time


def run(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--ID', type=int, default=0, help="yeah")

    args = parser.parse_args(argv[1:])

    my_ID = args.ID

    print("in the script: to_be_launched, ID: {}....sleeping".format(my_ID))
    time.sleep(5)
    print("in the script: to_be_launched, ID: {}...done sleeping".format(my_ID))


if __name__ == "__main__":
    run(sys.argv)
