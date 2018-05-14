from data import LastFm, Billboard
import csv


# with open('data/billboard_lyrics_1964-2015.csv', 'r') as csvfile:
#     songLyrics = csv.reader(csvfile, delimiter=',', quotechar='"')
#     lyrics = [row for row in songLyrics]

# print(len(lyrics))

api = LastFm()


# bb = Billboard()

# html = bb.get_song_lyrics('Never be like you', 'Flume featuring Kai')
# print(html)

destination =  open('data/billboard_lyrics_1964-2017_tagged.csv', 'a', encoding = 'utf-8')
originalData = [line for line in open('data/billboard_lyrics_1964-2017.csv', 'r',  encoding = 'utf-8')][1:]
songs = csv.reader(originalData, delimiter=',', quotechar='"' )



for i, row in enumerate(songs):

    artist = row[2]
    track = row[1]
    data = api.get_track_toptags(track, artist)

    line= ''
    if 'toptags' not in data:
        tags = []
    else:
        tags = data['toptags']['tag']
    if len(tags) > 5:
       tags = tags[:5]

    for tag in tags:
        line = line + '"'+tag['name']+'",'
        
    if len(tags) < 5:
        for j in range(5 - len(tags)):
            line += "NA,"
    
    line = originalData[i].strip()+',' +line[:-1] + '\n'
    destination.write(line)

