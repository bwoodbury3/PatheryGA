import urllib2
import json

# this file pulls a level from pathery and parses it into an array that the patheryai can understand

def getLevel(levelID):
    raw = urllib2.urlopen("http://pathery.com/a/map/" + str(levelID) + ".js").read()
    data = json.loads(raw)
    mapdata = data.get('tiles')
    mapcode = data.get('code')
    width = int(data.get('width'))
    height = int(data.get('height'))
    walls = int(data.get('walls'))
    teleporters = False
    level = []
    for i in range(height):
        row_data = mapdata[i]
        row = []
        for cell in row_data:
            if cell[0] == 'o':
                row.append(0)
            elif cell[0] == 'r':
                row.append('X')
            elif cell[0] == 'c':
                if cell[1] == '':
                    row.append('a')
                elif cell[1] == 2:
                    row.append('b')
                elif cell[1] == 3:
                    row.append('c')
                elif cell[1] == 4:
                    row.append('d')
                elif cell[1] == 5:
                    row.append('e')
            elif cell[0] == 't':
                row.append(0)
                print "Teleporter found....."
                teleporters = True
            elif cell[0] == 's':
                row.append('s')
            elif cell[0] == 'f':
                row.append('f')
        level.append(row)
    return (walls, level, mapcode, teleporters)
