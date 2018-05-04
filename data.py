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







