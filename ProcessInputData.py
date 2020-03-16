#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 14:46:05 2020

@author: cantus
"""

import codecs
import re
import numpy as np
import pandas as pd
from html.parser import HTMLParser

class MyHtmlParser(HTMLParser):
    def handle_data(self, data):
        pattern = re.compile(r"[\s]+")
        if pattern.match(data) == None:
            self.featureList = [float(item) for item in data.split()]
            
parser          = MyHtmlParser()
colonTissueData = None
f               = codecs.open("shuju.html","r")
lines           = f.readlines()
n               = 0
csvOutputPath   = r"/Users/cantus/Desktop/colonCancerData.csv"

for line in lines:
    n = n + 1
    
    if n >= 10 and n <= 2009:
        parser.feed(line)
        featureArray = np.array(parser.featureList)
        
        if n == 10:
            colonTissueData = featureArray
            colonTissueData = colonTissueData[:, None]
        else:
            colonTissueData = np.concatenate((colonTissueData, featureArray[:, None]), axis = 1)
        
colonTissueDf = pd.DataFrame(colonTissueData)
colonTissueDf.to_csv(csvOutputPath)









