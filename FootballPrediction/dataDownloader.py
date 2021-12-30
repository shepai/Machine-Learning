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
    def __init__(self,name="playerData.xml",Doc_360="360Data.json",show=True,Type='pass',maxData=10):
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
                        if i>maxData:
                            break
            except KeyError: #stop errors by ignoring them
                    pass
            if k>maxData:
                break
        print(len(tableOfData))
        self.tableOfData=tableOfData
        self.shortAverage=0
        self.mediumAverage=0
        self.longAverage=0
        self.players={}
        self.weightings={}
        self.values={}
    def sort(self,zone=[],direction=None,body_part=None,Type=None,pressureCare=None):
        #rewrite all data
        self.shortAverage=0
        self.mediumAverage=0
        self.longAverage=0
        self.players={}
        players={}
        self.weightings={}
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
                through_ball=row['through_ball']
                cross=row['cross']
                cut=row['cut_back']
                switch=row['switch']
                proceed=True
                if (pressureCare!=None): #check parameters are met
                    if pressure==True and pressureCare==True:
                        proceed=True
                    elif pressureCare==False and pressure==None:
                        proceed=True
                    else:
                        proceed=False
                if body_part!=None: #check body part match is needed
                    if body_part!=body:
                        proceed=False
                if Type!=None: #check type match is needed
                    if not(Type=='through_ball' and through_ball==True):
                        proceed=False
                    elif not(Type=='cross' and cross==True):
                        proceed=False
                    elif not(Type=='cut_back' and cut==True):
                        proceed=False
                    elif not(Type=='switch' and switch==True):
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
        #calculate overall averages
        if values['short']['kick']>0:
            self.shortAverage=values['short']['success']/values['short']['kick']
        if values['medium']['kick']>0:
            self.mediumAverage=values['medium']['success']/values['medium']['kick']
        if values['long']['kick']>0:
            self.longAverage=values['long']['success']/values['long']['kick']
        self.values={'short':values['short']['kick'],'medium':values['medium']['kick'],'long':values['long']['kick']}
        
        #calculate player averages
        self.weightings=players
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
    def getPlayerPerformance(self,playerName,view=True,ndig=1):
        assert self.players.get(playerName,False)!=False, "Player does not match the sort, so is not in the data"
        player=self.players[playerName]
        #gather weightings for player
        weight=self.weightings[playerName]
        shortPasses=weight['short']['kick']
        medPasses=weight['medium']['kick']
        longPasses=weight['long']['kick']
        all=shortPasses+medPasses+longPasses
        #weight=passes/allpasses
        shortWeight=shortPasses/all
        medWeight=medPasses/all
        longWeight=longPasses/all

        #gather weighting's for all
        shortPasses=self.values['short']
        medPasses=self.values['medium']
        longPasses=self.values['long']
        all=shortPasses+medPasses+longPasses
        #weight=passes/allpasses
        shortAllWeight=shortPasses/all
        medAllWeight=medPasses/all
        longAllWeight=longPasses/all

        averageP=round((((player['short']*shortWeight)+
                                   (player['medium']*medWeight)+
                                    (player['long']*longWeight))/3)*100,ndigits=ndig)
        averageA=round((((self.shortAverage*shortAllWeight)+
                                   (self.mediumAverage*medAllWeight)+
                                    (self.longAverage*longAllWeight))/3)*100,ndigits=ndig)
        if view: #output if wanted
            print(playerName,"performance")
            
            print("short Xpass:",round(100*self.shortAverage,ndigits=ndig),"%")
            print("player short actual:",round(100*player['short'],ndigits=ndig),"%")
            print("result",round((player['short']-self.shortAverage)*100,ndigits=ndig),"%")
            
            print("medium Xpass:",round(100*self.mediumAverage,ndigits=ndig),"%")
            print("player medium actual:",round(player['medium']*100,ndigits=ndig),"%")
            print("result",round((player['medium']-self.mediumAverage)*100,ndigits=ndig),"%")
            
            print("long Xpass:",round(100*self.longAverage,ndigits=ndig),"%")
            print("player long actual:",round(100*player['long'],ndigits=ndig),"%")
            print("result",round((player['long']-self.longAverage)*100,ndigits=ndig),"%")
            
            print("average performance",averageP-averageA,"%")
        return averageP-averageA
    def sortPlayers(self,direction=None,body_part=None,Type=None,pressureCare=None):
        #sort all the players based on parameters from worst to best
        self.sort(direction=None,body_part=None,Type=None,pressureCare=None)
        scores={}
        for player in self.players:
            score=self.getPlayerPerformance(player,view=False)
            scores[player]=score
        players={k: v for k, v in sorted(scores.items(), key=lambda item: item[1])}
        return players
    def get_feet(self):
        self.sort(        
d=downloader()
d.sort()
#'Jordan Brian Henderson'
print("All data\n")
d.getPlayerPerformance('Jordan Brian Henderson')

d.sortPlayers()

"""
Example:
Maxime Pelican Forward
['Maxime', 'Pelican', 'France', '1998-05-12', '65', '178', 'Unknown', 'Unknown', 'Unknown', '2016-07-01', '2017-01-09', 'France U20', 'France']

"""
