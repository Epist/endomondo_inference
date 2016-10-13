"""
This file needs to read in the json data and return both the structured data records for each data point as well as the
locations of each data point in the file
It also needs to be able to return a data point given a the known file position of that data point"""

import json

def getDataIndices(dataFileName):
    """Takes a data file name and scans through the file, recording the beginning and end file index for each data point
     Then returns this information as a list"""

    dataFile = open(dataFileName, 'r')
    dataIndices=[]# A list of (beginning, end) tuples
    eof=False
    curlyDepth=0 #A variable used for keeping track of beginnings and ends of data points
    currentIndices=[]#Placeholder
    numDataPointsAccessed=0
    #dataGen=fileIterator(dataFile, 10000)
    while not eof:
        if (numDataPointsAccessed==0) or (curlyDepth>0):
            #Go through the characters and keep track of the outermost brackets enclosing data points. Take note of their positions.
            nextChar=dataFile.read(1)
            #nextChar=dataGen.next()
            if nextChar == '{':
                if curlyDepth==1:
                    currentIndices.append(dataFile.tell()-1)
                    numDataPointsAccessed+=1
                curlyDepth=curlyDepth+1
            elif nextChar== '}':
                if curlyDepth==2:
                    #if numDataPointsAccessed%1000==0:
                    #    print numDataPointsAccessed
                    currentIndices.append(dataFile.tell())
                    dataIndices.append(currentIndices)
                    currentIndices=[]
                curlyDepth=curlyDepth-1
        else:
            eof = True

    return dataIndices

def getDataPoint(index, dataFile):
    """Takes a pair of data point indices (the beginning and end of the file location) and a file handle
    and reads the file between those positions, converting the json formatted data to a Python dictionary"""

    dataFile.seek(index[0])
    rawJson = dataFile.read(index[1]-index[0])

    #Convert the json into dictionaries
    dataPoint = json.loads(rawJson)

    return dataPoint

def fileIterator(dataFile, readSize):
    eof=False
    while eof==False:
        try:
            dataBlock=dataFile.read(readSize)
            for dp in dataBlock:
                yield dp
        except:
            eof=True
            
        
