import dsargparse

from . import train

def run(prog='python3 -m src'):
    parser = dsargparse.ArgumentParser(main=run, prog=prog)
    subparsers =  parser.add_subparsers(help='command')
    subparsers.add_parser(train.train, add_arguments_auto=True)
    return parser.parse_and_run()