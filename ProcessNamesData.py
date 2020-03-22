import codecs
import re
import numpy as np
import pandas as pd
from html.parser import HTMLParser

colonGeneDescription     = None
colonGeneDescriptionData = None
f                        = codecs.open("names.html", "r")
lines                    = f.readlines()
n                        = 0
csvOutputPath            = r"./colonGeneDescriptions.csv"
p1                       = re.compile(r"([\w\.]+)\s([\w]+)\s(3' UTR)\s([\w]+)\s([\w]+)\s(.+)")
p2                       = re.compile(r"([\w\.]+)\s([\w]+)\s(gene)\s([\w]+)\s(.+)")

for line in lines:
    n = n + 1
    
    if n >= 11 and n <= 2010:
        m1 = p1.match(line)
        m2 = p2.match(line)
        
        if m1 != None:
            colonGeneDescription = np.array([m1.group(i + 1).strip() for i in range(6)])
            
        elif m2 != None:
            colonGeneDescription = np.array([m2.group(1).strip(),
                                             m2.group(2).strip(),
                                             m2.group(3).strip(),
                                             m2.group(4).strip(),
                                             'N/A',
                                             m2.group(5).strip()])
        else:
            m3 = line.split()
            colonGeneDescription = np.array([m3[0].strip(),
                                             "N/A",
                                             "N/A",
                                             "N/A",
                                             'N/A',
                                             m3[1].strip()])
            
        if n == 11:
            colonGeneDescriptionData = colonGeneDescription
            colonGeneDescriptionData = colonGeneDescriptionData[:, None]
            
        else:
            colonGeneDescriptionData = np.concatenate((colonGeneDescriptionData, colonGeneDescription[:, None]), axis = 1)

colonTissueDf = pd.DataFrame(colonGeneDescriptionData.T)
colonTissueDf.to_csv(csvOutputPath)
