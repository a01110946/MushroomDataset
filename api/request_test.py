import http.client
import json

conn = http.client.HTTPConnection("localhost", 8000)
payload = json.dumps({
    "cap_diameter": 5.2,
    "cap_surface": "s",
    "cap_color": "n",
    "gill_attachment": "f",
    "stem_width": 8.3,
    "stem_root": "c",
    "stem_surface": "s",
    "veil_type": "p",
    "veil_color": "w",
    "has_ring": "t",
    "spore_print_color": "k"
})
headers = {'Content-Type': 'application/json'}
conn.request("POST", "/predict", payload, headers)
response = conn.getresponse()
print(response.status, response.reason)
print(response.read().decode())