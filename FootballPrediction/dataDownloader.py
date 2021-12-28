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
                               struct={}
                               for i,dat in enumerate(player.findall("Stat")):
                                   struct[dat.attrib['Type']]=dat.text
                               print(struct)
                                    
                   
downloader()


"""
Example:
Maxime Pelican Forward
['Maxime', 'Pelican', 'France', '1998-05-12', '65', '178', 'Unknown', 'Unknown', 'Unknown', '2016-07-01', '2017-01-09', 'France U20', 'France']

"""
