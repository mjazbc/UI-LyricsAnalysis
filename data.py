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
        if time.time() - self.lastcall < 1:
            time.sleep(time.time() - self.lastcall)

        param = {'method': method, 'track':track, 'artist':artist, 'api_key' : self.apiKey, 'format' : 'json', 'autocorrect':1}
        r = requests.get(self.lastFmUrl, params=param).json()

        self.lastcall = time.time()

        return r

class Billboard:

    wikiUrl = 'https://en.wikipedia.org/wiki/Billboard_Year-End_Hot_100_singles_of_{}'
    metroLyricsRootUrl = 'http://www.metrolyrics.com/{0}'
    metroLyricsSearch = 'http://api.metrolyrics.com/v1//multisearch/all/X-API-KEY/196f657a46afb63ce3fd2015b9ed781280337ea7/format/json?find={}'

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
        return '\n'.join(verse.text for verse in lyricsNode.find_all('p', {'class' : 'verse'}))

    def search_lyrics(self, track, artist):
        track = track.replace('-', '+').lower()
        artist = artist.replace('-', '+').lower()
        r = requests.get(self.metroLyricsSearch.format(track+'+'+artist))

        jsonResult = json.loads(r.text)
        url = jsonResult['results']['lyrics']['d'][0]['u']

        lyricsDiv = self.get_lyrics_div(url)

        return self.extract_lyrics(lyricsDiv)

    def get_lyrics_div(self, url):
        metroUrl = str.format(self.metroLyricsRootUrl, url)
        r = requests.get(metroUrl)
        soup = BeautifulSoup(r.text, 'html.parser')
        return soup.find('div', {'id':'lyrics-body-text'})

    def get_song_lyrics(self, track, artist):
        track = track.replace(' ', '-').lower()
        artist = artist.replace(' ', '-').lower()
        
        lyricsDiv = self.get_lyrics_div(track+'-lyrics-'+ artist+'.html')

        #Try finding lyrics by url
        if lyricsDiv:
            return self.extract_lyrics(lyricsDiv)

        #If url doesn't exist, try search
        return self.search_lyrics(track, artist)
    







