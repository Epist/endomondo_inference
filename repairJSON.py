#Repairs the json format in te Endomondo data crawls

import string 

newFile = open('endomondoHR_proper.json','w+')
newFile.write('{\n "users" : [ \n')
with open("endomondoHR.json") as infile:
    for line in infile:
        normLine=line.decode('ascii')
        transline=string.replace(normLine, "'", '"')
        transline=string.replace(transline, "}", '},')
        newFile.write(transline)

newFile.seek(-2,2)
newFile.write(']\n}')
newFile.close()