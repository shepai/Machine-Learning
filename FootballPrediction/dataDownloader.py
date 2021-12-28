"""
Football data downloader

Using the open data from Statsbomb

Code by Dexter R Shepherd, aged 20
"""

from xml.etree import ElementTree as ET

class downloader:
    def __init__(self,name="360_data.xml"):
        xml = ET.parse(name)
        root_element = xml.getroot()
        for child in root_element:
            print(self.xml_dict(child))
    def xml_dict(self,node, path="", dic =None):
        if dic == None:
            dic = {}
        name_prefix = path + ("." if path else "") + node.tag
        numbers = set()
        for similar_name in dic.keys():
            if similar_name.startswith(name_prefix):
                numbers.add(int (similar_name[len(name_prefix):].split(".")[0] ) )
        if not numbers:
            numbers.add(0)
        index = max(numbers) + 1
        name = name_prefix + str(index)
        dic[name] = node.text + "<...>".join(childnode.tail
                                             if childnode.tail is not None else
                                             "" for childnode in node)
        for childnode in node:
            self.xml_dict(childnode, name, dic)
        return dic
downloader()
