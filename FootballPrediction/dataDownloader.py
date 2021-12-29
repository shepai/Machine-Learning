"""
Football data downloader

Using the open data from Statsbomb

Code by Dexter R Shepherd, aged 20
"""

from xml.etree import ElementTree as ET
import json
from matplotlib import pyplot as plt
import statsbomb as sb
import pandas as pd
import math as maths

class downloader:
    def __init__(self,name="playerData.xml",Doc_360="360Data.json",show=True,Type='pass'):
        comps = sb.Competitions()
        json_data = comps.data
        df = comps.get_dataframe()
        keys=list(df.keys())
        #competitions
        tableOfData=[]
        for k,competition in enumerate(json_data):
            id=competition['competition_id']
            season_id=competition['season_id']
            #print(competition['match_updated_360'])
            try:
                    matches_ = sb.Matches(event_id=id,season_id=season_id)
                    df_match = matches_.get_dataframe()
                    for i in range(len(df_match['match_id'])):
                        #gather information about this match
                        matchID=df_match['match_id'][i]
                        matchDate=df_match['match_date'][i]
                        season=df_match['season'][i]
                        comp=df_match['competition'][i]
                        homeS=df_match['home_score'][i]
                        awayS=df_match['away_score'][i]
                        stadium=df_match[ 'stadium'][i]['name']
                        #print(stadium,matchDate,homeS,awayS)
                        #_360Table.append([competition['match_available_360'],
                                          #matchID,matchDate,season,comp,homeS,
                                          #awayS,stadium]) #create table
                        
                        events = sb.Events(event_id=str(matchID))
                        df = events.get_dataframe(event_type=Type)
                        tableOfData.append(df)
                        ##print(len(df))  # 23
                        if i>10:
                            break
            except KeyError: #stop errors by ignoring them
                    pass
            if k>5:
                break
        print(len(tableOfData))
        self.tableOfData=tableOfData
        self.shortAverage=0
        self.mediumAverage=0
        self.longAverage=0
        self.players={}
    def sort(self,zone=[],direction=None,body_part=None,Type=None,pressureCare=None):
        players={}
        values={'short':{'kick':0,'success':0},'long':
                    {'kick':0,'success':0},'medium':{'kick':0,'success':0}}
        SHORT=15 #short up to
        MED=40 #medium up to
        for data in self.tableOfData:
            df=pd.DataFrame(data)
            for i, row in df.iterrows():
                #gather data
                
                name=row['player']
                pressure=row['under_pressure']
                outcome=row['outcome']
                body=row['body_part']
                proceed=True
                if (pressureCare!=None): #check parameters are met
                    if pressure==True and pressureCare==True:
                        proceed=True
                    elif pressureCare==False and pressure==None:
                        proceed=True
                    else:
                        proceed=False
                if body_part!=None:
                    if body_part!=body:
                        proceed=False
                if proceed: #if parameters are met
                    if outcome!='Injury Clearance' and outcome!='Pass Offside':
                        
                        x1=row['start_location_x']
                        y1=row['start_location_y']
                        x2=row['end_location_x']
                        y2=row['end_location_y']
                        zone=None
                        dist=int(maths.sqrt((x1-x2)**2 +(y1-y2)**2)) #calculate distance
                        size=""
                        if dist<=SHORT: #gathers the size parameter
                            size='short'
                        elif dist<=MED:
                            size='medium'
                        elif dist>MED:
                            size='long'
                        successValue=0
                        if outcome not in ['Incomplete','Out']:
                            successValue=1
                        if players.get(name,False)==False: #score
                            players[name]={'short':{'kick':0,'success':0},'long':
                                           {'kick':0,'success':0},'medium':{'kick':0,'success':0}}
                            players[name][size]={'kick':1,'success':successValue}
                        else: #increase
                            tmp=players[name][size]
                            tmp['kick']+=1
                            tmp['success']+=successValue
                            players[name][size]=tmp
                        #store for general population
                        tmp=values[size]
                        tmp['kick']+=1
                        tmp['success']+=successValue
                        values[size]=tmp
        
        self.shortAverage=values['short']['success']/values['short']['kick']
        self.mediumAverage=values['medium']['success']/values['medium']['kick']
        self.longAverage=values['long']['success']/values['long']['kick']
        
        for player in players: #prganise performances
            short=players[name]['short']
            med=players[name]['medium']
            long=players[name]['long']
            self.players[player]={'short':0,'long':
                                       0,'medium':0}
            if short['kick']>0:
                self.players[player]['short']=short['success']/short['kick']
            if med['kick']>0:
                self.players[player]['medium']=med['success']/med['kick']
            if long['kick']>0:
                self.players[player]['long']=long['success']/long['kick']
    def getPlayerPerformance(self,playerName,view=True):
        assert self.players.get(playerName,False)!=False, "Player does not exist"
        player=self.players[playerName]
        if view: #output if wanted
            print(playerName,"performance")
            
            print("short Xpass:",self.shortAverage)
            print("player short Xpass:",player['short'])
            print("result",player['short']-self.shortAverage)
            
            print("medium Xpass:",self.mediumAverage)
            print("player medium Xpass:",player['medium'])
            print("result",player['medium']-self.mediumAverage)
            
            print("long Xpass:",self.longAverage)
            print("player long Xpass:",player['long'])
            print("result",player['long']-self.longAverage)
        return player['short']-self.shortAverage,player['medium']-self.mediumAverage,player['long']-self.longAverage

d=downloader()
d.sort(body_part='Right Foot',pressureCare=True)
#'Jordan Brian Henderson'
print("Under pressure\n")
d.getPlayerPerformance('Jordan Brian Henderson')
d.sort()
print("All pressure\n\n")
#'Jordan Brian Henderson'
d.getPlayerPerformance('Jordan Brian Henderson')


"""
Example:
Maxime Pelican Forward
['Maxime', 'Pelican', 'France', '1998-05-12', '65', '178', 'Unknown', 'Unknown', 'Unknown', '2016-07-01', '2017-01-09', 'France U20', 'France']

"""
