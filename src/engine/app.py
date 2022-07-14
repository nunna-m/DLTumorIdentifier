"""
Tumor Identifier CLI

Usage:
    app train <temp>
    app train_stacked <temp>
    app -h|--help

Options:
    <temp> pass in any string. echo.
"""


from docopt import docopt
import train, train_stacked

def run():
    arguments = docopt(__doc__, version='DEMO 1.0')
    print(arguments)
    if arguments['-h']:
        print(train.train.__doc__)
    if arguments['train']:
        print('train')
        train.train(arguments['<temp>'])
        
    # if arguments['train_stacked']:
    #     print('train_stacked')
    #     train_stacked.train(arguments['<another_temp>'])

run()






# import dsargparse

# from . import train

# def run(prog='python3 -m src'):
#     parser = dsargparse.ArgumentParser(main=run, prog=prog)
#     subparsers =  parser.add_subparsers(help='command')
#     subparsers.add_parser(train.train, add_arguments_auto=True)
#     return parser.parse_and_run()