import dsargparse


from . import dataClass
from . import generateData
from . import modalityStack
from . import newCrossVal
from . import visualizeNumpy
from . import aggregate

def main(prog='python3 -m src.data'):
    parser = dsargparse.ArgumentParser(main=main, prog=prog)
    subparsers = parser.add_subparsers(help='command')
    subparsers.add_parser(aggregate.createRaw, add_arguments_auto=True)
    subparsers.add_parser(aggregate.createNumpy, add_arguments_auto=True)
    subparsers.add_parser(visualizeNumpy.visualizeSample, add_arguments_auto=True)
    return parser.parse_and_run()

if __name__ == '__main__':
    main()