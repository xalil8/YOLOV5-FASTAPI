''' Example client sending POST request to server and printing the YOLO results
'''
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import requests as r
import json
from pprint import pprint

def send_request(image = '/Users/halil/Desktop/gity/images/berk_data.jpeg', model_name = 'berk'):

    
    res = r.post("http://localhost:8000", 
                data={'model_name': model_name}, 
                files = {'file': open(image , "rb")})     #pass the files here

    pprint(json.loads(res.text))

if __name__ == '__main__':
    send_request()