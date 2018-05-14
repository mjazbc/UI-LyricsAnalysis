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

    print(tc.get_genre(tags))
    line= ''
    
    line = originalData[i].strip()+',' +line[:-1] + '\n'
    destination.write(line)

