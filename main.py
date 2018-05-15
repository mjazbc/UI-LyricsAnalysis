from data import LastFm, Billboard, TagClassifier
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
originalData = [line for line in open('data/billboard_lyrics_1964-2017_tagged.csv', 'r',  encoding = 'utf-8')][1:]
songs = csv.reader(originalData, delimiter=',', quotechar='"' )


tc = TagClassifier()

for i, row in enumerate(songs):

    tags = row[6:12]
    era =''
    year = int(row[3])
    
    tag = 'other'
    if all(tag == 'NA' for tag in tags):
        tag = 'NA'
    else:
        g = tc.get_genre(tags)
        if g:
            tag = g

    
    if year < 1970:
        era = '60s'
    elif year < 1980:
        era = '70s'
    elif year < 1990:
        era = '80s'
    elif year < 2000:
        era = '90s'
    elif year < 2010:
        era = '2000s'
    elif year < 2020:
        era = '2010s'
        
    
    line = originalData[i].strip()+',' +tag + ',' +era + '\n'
    destination.write(line)

