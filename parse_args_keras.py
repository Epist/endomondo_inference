import sys, argparse

def parse_args_keras(argv, model):
    
    #if there is an argument specified for error metric, attributes, or drop in, overwrite the defaults for the corresponding variables to the values specified in the arguments
    #also print the new values and the fact that they were changed for confirmation
    
    parser = argparse.ArgumentParser(description='Specify some model parameters')
    
    #parser.add_argument('-di', dest='drop_in', action='store_true', default=False, help='Use dropin')
    #parser.add_argument('-em', dest='lossType', action='store', help='Specify the error metric')
    parser.add_argument('-a', dest='attributes', action='store', nargs='+', help='Specify the attributes')
    parser.add_argument('-fn', dest='fileNameEnding', action='store', help='Append an identifying string to the end of output files')
    parser.add_argument('-t', dest='target_atts', action='store', nargs='+', help='Specify the target attributes')
    parser.add_argument('-sc', dest='scale_toggle', action='store', default="True", help='Should the data be scaled and normalized before entering the model?')
    
    
    args = parser.parse_args()
    
    print(args)
    
    #if args.drop_in:
    #    dropInEnabled = True
    #    model = "FixedDropin"
    #    print("Toggled drop in from command line")
        
    #try:
    #if args.lossType == "RMSE" or args.lossType == "MAE":
    #    lossType = args.lossType
    #    print("Added loss type " + args.lossType + " from command line")
    #else:
    #    print(args.lossType + " is not a valid loss type. Reverting to " + lossType)
    #except:
    #    pass
    
    #try:
    
    if args.target_atts is not None:
        model.targetAtts = args.target_atts
        print("Added targets from command line: " + str(model.targetAtts))
    
    if args.attributes is not None:
        model.endoFeatures = args.attributes
        model.inputOrderNames = [x for x in model.endoFeatures if x not in model.targetAtts]
        print("Added attributes from command line: " + str(model.endoFeatures))
    
    if args.fileNameEnding is not None:
        model.model_file_name = args.fileNameEnding

    if args.scale_toggle == "True" or "true":
        model.scale_toggle = True
        print("Scaling the data")
    elif args.scale_toggle == "False" or "false":
        model.scale_toggle = False
        print("Not scaling the data")
    else:
        raise(exception("Scale toggle paramater must be either true or false"))
        

