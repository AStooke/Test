
import sys
import argparse


def func(first, second):
    print(first, type(first))
    print(second, type(second))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("first")
    parser.add_argument("second")
    parser.add_argument("positionals", nargs="*")
    args = parser.parse_args(sys.argv[1:])
    print(args.positionals)
    func(args.first, args.second, *args.positionals)

