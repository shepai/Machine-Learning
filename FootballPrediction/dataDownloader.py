"""
Football data downloader

Using the open data from Statsbomb

Code by Dexter R Shepherd, aged 20
"""

from xml.etree import ElementTree as ET

class downloader:
    def __init__(self,name="360_data.xml"):
        tree = ET.parse(name)
        root = tree.getroot()
        self.struct={}
        print(root.findall('SoccerDocument'))
        for graph in root.findall('SoccerDocument'):
                   for g in graph.findall('PlayerChanges'): #check the words saved
                       for team in g.findall('Team'):
                           for player in team.findall('Player'):
                               name=player.find('Name').text
                               position=player.find('Position').text
                               print(name,position)
                               data=[]
                               for i in player.findall("Stat"):
                                   data.append(i.text)
                               print(data)
                                    
                   
downloader()
