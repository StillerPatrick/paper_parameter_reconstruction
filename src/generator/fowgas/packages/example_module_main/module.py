def hello():
    """
    Hello world.
    """
    print('Hello world')



import sys
import getopt


class Usage(Exception):
    def __init__(self, msg):
        self.msg = msg

def main(argv=None):
    if argv is None:
        argv = sys.argv
    print('argv: ', argv)
    try:
        try:
            opts, args = getopt.getopt(argv[1:], 'a:h', ['help',])
            print('opts: ', opts)
            print('args: ', args)
        except getopt.error as msg:
            raise Usage(msg)
    except Usage as err:
        print(err.msg, file=sys.stderr)
        print('For help use --help', file=sys.stderr)
        return 2


if __name__=='__main__':
    sys.exit(main())
