def parse_args_keras(argv):
    
    #if there is an argument specified for error metric, attributes, or drop in, overwrite the defaults for the corresponding variables to the values specified in the arguments
    #also print the new values and the fact that they were changed for confirmation
    
    parser = argparse.ArgumentParser(description='Specify some model parameters')
    
    #parser.add_argument('-di', dest='drop_in', action='store_true', default=False, help='Use dropin')
    #parser.add_argument('-em', dest='lossType', action='store', help='Specify the error metric')
    parser.add_argument('-a', dest='attributes', action='store', nargs='+', help='Specify the attributes')
    parser.add_argument('-fn', dest='fileNameEnding', action='store', help='Append an identifying string to the end of output files')
    parser.add_argument('-t', dest='target_att', action='store', help='Specify the target attribute')
    
    
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
    
    if args.target_att is not None:
        self.targetAtt = args.target_att
        print("Added target from command line: " + str(args.target_att))
    
    if args.attributes is not None:
        self.endoFeatures = args.attributes
        self.inputOrderNames = [x for x in endoFeatures if x!=targetAtt]
        print("Added attributes from command line: " + str(endoFeatures))
    
    if args.fileNameEnding is not None:
        self.model_file_name = args.fileNameEnding
        

