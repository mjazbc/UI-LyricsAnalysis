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

html = bb.get_song_lyrics('Never be like you', 'Flume featuring Kai')
print(html)