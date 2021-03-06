import requests
import time
import json
from bs4 import BeautifulSoup

class LastFm:

    apiKey = ''
    lastFmUrl = 'http://ws.audioscrobbler.com/2.0/'
    lastcall = 0

    def __init__(self):
        with open('secrets.json') as json_data:
            config = json.load(json_data)
            self.apiKey = config['LastFmApiKey']

    def get_track_data(self, track, artist):
         return self.call_api('track.getInfo', track, artist)
    
    def get_track_toptags(self, track, artist):        
        return self.call_api('track.getTopTags', track, artist)
    
    def call_api(self, method, track, artist):

        # Only 1 request per second should be made
        # if time.time() - self.lastcall < 1:
            # time.sleep(time.time() - self.lastcall)
        time.sleep(1.5)

        param = {'method': method, 'track':track, 'artist':artist, 'api_key' : self.apiKey, 'format' : 'json', 'autocorrect':1}
        r = requests.get(self.lastFmUrl, params=param).json()

        self.lastcall = time.time()

        return r

class Billboard:

    wikiUrl = 'https://en.wikipedia.org/wiki/Billboard_Year-End_Hot_100_singles_of_{}'
    metroLyricsRootUrl = 'http://www.metrolyrics.com/{0}'
    metroLyricsSearch = 'http://api.metrolyrics.com/v1//multisearch/all/X-API-KEY/196f657a46afb63ce3fd2015b9ed781280337ea7/format/json?find={}'
    geniusSearchUrl = 'https://genius.com/api/search/multi?q={0}'

    def get_song_list(self, year):
        yearUrl = self.wikiUrl.format(year)
        r = requests.get(yearUrl)
        soup = BeautifulSoup(r.text, 'html.parser')

        table = soup.find('table')
        songList = []

        for tr in table.find_all('tr'):
            row = [td.text.strip('"') for td in  tr.find_all('td')]
            if row:
                songList += [row]
        
        return songList

    def extract_lyrics(self, lyricsNode):
        return ' '.join(verse.text for verse in lyricsNode.find_all('p', {'class' : 'verse'}))

    def search_lyrics_metro(self, track, artist):
        track = track.replace('-', '+').lower()
        artist = artist.replace('-', '+').lower()

        if 'featuring' in artist:
            artist = artist[0:artist.index('featuring')-1]
        elif ' and ' in artist:
            artist = artist[0:artist.index('and')-1]

        r = requests.get(self.metroLyricsSearch.format(track+'+'+artist))

        jsonResult = json.loads(r.text)

        if(artist.replace('-', ' ') not in jsonResult['results']['lyrics']['d'][0]['p'].lower()):
            return
        url = jsonResult['results']['lyrics']['d'][0]['u']

        lyricsDiv = self.get_lyrics_div(url)

        return self.extract_lyrics(lyricsDiv)
    
    def search_lyrics_genius(self, track, artist):
        track = track.replace('-', '+').lower()
        artist = artist.replace('-', '+').lower()

        if 'featuring' in artist:
            artist = artist[0:artist.index('featuring')-1]
        elif ' and ' in artist:
            artist = artist[0:artist.index('and')-1]

        r = requests.get(self.geniusSearchUrl.format(track+'+'+artist))

        jsonResult = json.loads(r.text)

        if(artist.replace('+',' ') not in jsonResult['response']['sections'][0]['hits'][0]['result']['full_title'].replace(u'\xa0', u' ').lower()):
            return
        url = 'https://genius.com'+jsonResult['response']['sections'][0]['hits'][0]['result']['path']

        return self.extract_lyrics_genius(url)


    def get_lyrics_div(self, url):
        metroUrl = str.format(self.metroLyricsRootUrl, url)
        r = requests.get(metroUrl)
        soup = BeautifulSoup(r.text, 'html.parser')
        return soup.find('div', {'id':'lyrics-body-text'})
    
    def extract_lyrics_genius(self, url):
        r = requests.get(url)
        soup = BeautifulSoup(r.text, 'html.parser')
        lyricsDiv = soup.find('div', {'class':'lyrics'})

        lyrics = ' '.join([a.text for a in lyricsDiv.find_all('a') if a.text[0] != '['])

        return lyrics

    def get_song_lyrics(self, track, artist):
        track = track.replace(' ', '-').lower()
        artist = artist.replace(' ', '-').lower()

        artist = artist.replace('&', 'and')
        track = track.replace('&', 'and').replace('\'','')
        
        lyricsDiv = self.get_lyrics_div(track+'-lyrics-'+ artist+'.html')

        #Try finding lyrics by url
        if lyricsDiv:
            return self.extract_lyrics(lyricsDiv)

        #If url doesn't exist, try searching lyrics on MetroLyrics
        lyrics = self.search_lyrics_metro(track, artist)
        if lyrics:
            return lyrics
        
        #Try searcing on genius.com
        return self.search_lyrics_genius(track,artist)
    
class TagClassifier:

    genres = ['pop', 'rock', 'soul', 'hiphop', 'dance', 'country']

    pop_subgenres = ['female vocalists', 'pop rock']
    rock_subgenres = ['classic rock', 'garage rock', 'folk rock', 'rock n roll', 'progressive rock', 'alternative rock', 'heavy metal', 'punk rock', 'punk']
    soul_subgenres = ['jazz', 'motown', 'ballad', 'smooth jazz', 'rnb', 'r&b']
    hiphop_subgenres = ['hip hop', 'rap']
    dance_subgenres = ['disco', 'eurodance', 'house', 'electronic', 'trance', 'techno']
    country_subgenres = ['country music']

    mapping = [pop_subgenres, rock_subgenres, soul_subgenres, hiphop_subgenres, dance_subgenres, country_subgenres ]

    def get_genre(self, tagsList):
        for tag in tagsList:
            tag = self.sanitize(tag)

            if tag in self.genres:
                return tag
                
            generalized = self.generalize(tag)
            if generalized in self.genres:
                return generalized
        return

    def generalize(self, tag):
        for i, subgenresList in enumerate(self.mapping):
            if tag in subgenresList:
                return self.genres[i]

    def sanitize(self, string):
        return string.replace('-','').lower().strip()







