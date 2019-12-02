import sys, getopt


# this function is used to read and parse the parameters from the command line
def parse(args):
    parseargs = {}
    try:
        options, arguments = getopt.getopt(args, 'x:o:', ['run=', 'ver=', 'mem='])
        for opt, arg in options:
            # the run number - affects the folder name
            if opt == '--run':
                parseargs['run'] = int(arg)
            # the version number of a saved model
            if opt == '--ver':
                parseargs['ver'] = int(arg)
            # the version number of a instance of saved action values to replay to the model
            if opt == '--mem':
                parseargs['mem'] = int(arg)
            # specifies the player type of player x and o
            if opt == '-x':
                parseargs['x'] = arg
            if opt == '-o':
                parseargs['o'] = arg

        return parseargs
    except (getopt.GetoptError, ValueError) as e:
        print(str(e))
        print("There was an error encountered! Correct usage: \n "
              "-x : type of first player [a,h], a = AI, h = Human\n"
              "-o : type of second player [a,h], a = AI, h = Human\n"
              "--run : the run folder number\n"
              "--ver : the model version of the run folder \n"
              "--mem : the memory version of the run folder\n")
        sys.exit(2)

