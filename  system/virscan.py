import requests
import os
import json
from time import sleep
import rich

FILE = r"D:\DownLoad\vs_BuildTools.exe"
headers = {
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "cookie": "SERVER_ID=b32141afb3592f569a05cd715e456344; _ga=GA1.1.1643123218.1703328674; _ga_Y28NNXRNRZ=GS1.1.1703328673.1.1.1703328714.19.0.0"}

if __name__ == '__main__':
    params = {
        "filename": os.path.basename(FILE),
        "size": str(os.path.getsize(FILE))
    }
    session = requests.Session()
    session.headers = headers
    r = session.post("https://www.virscan.org/v1/file/sha256/upload", params=params, files={"file": open(FILE, 'rb')})
    print(r.text)
    sha256 = json.loads(r.text)['data']['sha256']
    while True:
        r = session.get(f"https://www.virscan.org/v1/file/sha256/search/{sha256}?page=detail")
        rich.print_json(r.text)
        sleep(2)
