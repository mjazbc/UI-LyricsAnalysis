from data import LastFm, Billboard
import csv


# with open('data/billboard_lyrics_1964-2015.csv', 'r') as csvfile:
#     songLyrics = csv.reader(csvfile, delimiter=',', quotechar='"')
#     lyrics = [row for row in songLyrics]

# print(len(lyrics))

api = LastFm()
# data = api.get_track_toptags('american idiot', 'green day')

# for tag in data['toptags']['tag'][:3]:
#     print(tag['name'])


bb = Billboard()

# html = bb.get_song_lyrics('Never be like you', 'Flume featuring Kai')
# print(html)

destination = open('data/newlyrics.csv', 'a', encoding = 'utf-8')
with open('data/newsongs.csv', 'r') as csvfile:
    songs = csv.reader(csvfile, delimiter=',', quotechar='"')
    for i, row in enumerate(songs):
        if i < 116:
            continue

        artist = row[3]
        track = row[2]
        lyrics = bb.get_song_lyrics(track,artist)

        if not lyrics:
            lyrics = "NA"
        line = row[1]+',"'+track+'","'+artist+'",'+row[0]+',"'+lyrics.replace('\"', '').replace('\n',' ')+'\n'

        destination.write(line)

