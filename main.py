from data import LastFm, Billboard, TagClassifier
import csv
import wordninja


# with open('data/billboard_lyrics_1964-2015.csv', 'r') as csvfile:
#     songLyrics = csv.reader(csvfile, delimiter=',', quotechar='"')
#     lyrics = [row for row in songLyrics]

# print(len(lyrics))

api = LastFm()


# bb = Billboard()

# html = bb.get_song_lyrics('Never be like you', 'Flume featuring Kai')
# print(html)

originalData = [line for line in open('data/preprocessed.csv', 'r',  encoding = 'utf-8')][1:]
destination =  open('data/preprocessed_fixed.csv', 'a', encoding = 'utf-8')
songs = csv.reader(originalData, delimiter=';', quotechar='"' )

def fix_parsing_errors(songs, destination):
     for i, row in enumerate(songs):
        lyrics = row[4]
        splitWords = wordninja.split(lyrics)

        if 'miscellaneous' in splitWords:
            miscIndex = splitWords.index('miscellaneous')
            if miscIndex < 5:
                splitWords = splitWords[miscIndex +1 :]
        
        fixed = ' '.join(splitWords)
        # print(len(lyrics), ' ', len(fixed))

        line = originalData[i].replace(lyrics, fixed)
        destination.write(line)

def set_genres(songs, destination):
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


fix_parsing_fuckups(songs,destination)