import argparse
import os

import dsargparse


from . import trainvaltest
from . import crossVal

def main(prog='python3 -m src.preproc'):
    parser = dsargparse.ArgumentParser(main=main, prog=prog)
    subparsers = parser.add_subparsers(help='command')
    # subparsers.add_parser(trainvaltest.split, add_arguments_auto=True)
    # subparsers.add_parser(trainvaltest.remove_existing_folder, add_arguments_auto=True)
    subparsers.add_parser(crossVal.crossvalDataGen, add_arguments_auto=True)
    return parser.parse_and_run()

if __name__ == '__main__':
    main()