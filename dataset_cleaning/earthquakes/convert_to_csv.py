import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup

tree = ET.parse('earthquakes.xml')
columns = ['id', 'origin', 'magnitude', 'year', 'month', 'day', 'hour', 'minute', 'second', 'latitude', 'longitude', 'depth', 'region']
root = tree.getroot()[0]
#for row in root:
    #print(row.tag, row.attrib)
namespaces = {'catalog': 'http://anss.org/xmlns/catalog/0.1', 'event': 'event'} # add more as needed
ns = {'real_person': 'http://people.example.com',
      'role': 'http://characters.example.com'}
for event in root.findall('event:event', namespaces):
    #print(event.tag, event.attrib)
    event_id = event.get('{http://anss.org/xmlns/catalog/0.1}eventid')
    counter = 0
    for origin in event.findall('event:origin', namespaces):
        time = origin.find('./event:time', namespaces)#.text#find('event:value')#[0].findtext('event:value', namespaces)
        for s in origin:
            print(s)
        #if time == None:
        #    print(origin.tag, origin.attrib)
        #    print(event_id)
        #    continue
        counter += 1
    for magnitued in event.findall('event:magnitude', namespaces):
        pass
    #print(event.get("{http://anss.org/xmlns/catalog/0.1}eventsource"))
    #print(event.find("event:description", namespaces))
    #print(event.tag, event.attrib)

    #event_id = 


