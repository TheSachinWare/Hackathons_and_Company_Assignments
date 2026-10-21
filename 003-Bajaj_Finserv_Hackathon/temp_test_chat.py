import urllib.request
import urllib.error
import json

url = 'http://127.0.0.1:8000/chat'
data = json.dumps({'message': 'How much EMI do I pay each month?'}).encode('utf-8')
req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
try:
    with urllib.request.urlopen(req, timeout=10) as resp:
        print('status', resp.status)
        print(resp.read().decode())
except urllib.error.HTTPError as e:
    print('http error', e.code)
    print(e.read().decode())
except Exception as e:
    print('error', type(e).__name__, e)
