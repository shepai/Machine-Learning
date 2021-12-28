"""
Football data downloader

Using the open data from Statsbomb

Code by Dexter R Shepherd, aged 20
"""

from xml.etree import ElementTree as ET
import json
from matplotlib import pyplot as plt

class downloader:
    def __init__(self,name="playerData.xml",Doc_360="360Data.json",show=True):
        tree = ET.parse(name)
        root = tree.getroot()
        self.struct={}
        #gather player information
        for graph in root.findall('SoccerDocument'):
                   for g in graph.findall('PlayerChanges'): #check the words saved
                       for team in g.findall('Team'):
                           for player in team.findall('Player'):
                               name=player.find('Name').text
                               position=player.find('Position').text
                               #print(name,position)
                               struct={}
                               for i,dat in enumerate(player.findall("Stat")):
                                   struct[dat.attrib['Type']]=dat.text
                               if self.struct.get(player,False)==False:
                                   #add if not in array
                                   self.struct[player]=struct
                               else: #traceback output
                                   print("found duplicate data for", name, "keeping first")
                                   print(self.struct[player])
        #create table
        self.TABLE=[]
        events={}
        #TABLE entry [player, zone, pressure, distance of shot, success, foot]      
        #gather 360 data
        file=open(Doc_360) #read file
        r=file.read()
        file.close()
        moves = json.loads(r) #convert to dictionary
        for event in moves:
            id=event['event_uuid']
            area=event['visible_area']
            people=event['freeze_frame']
            loc=[]
            for person in people:
                coord=person['location']
                if person['actor']: #player with ball
                    loc=coord
                    plt.scatter(coord[0],coord[1],c="y")
                elif person['teammate']: #teammate
                    plt.scatter(coord[0],coord[1],c="g")
                else: #enemy
                    plt.scatter(coord[0],coord[1],c="r")
                
            if show: #only show if parameter allows
                plt.xlim([0, 100])
                plt.ylim([0, 100])
                plt.show()
            events[id]=loc
            

       
                                    
                   
downloader()


"""
Example:
Maxime Pelican Forward
['Maxime', 'Pelican', 'France', '1998-05-12', '65', '178', 'Unknown', 'Unknown', 'Unknown', '2016-07-01', '2017-01-09', 'France U20', 'France']

"""
