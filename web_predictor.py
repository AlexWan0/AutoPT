import requests
import json
from labels import labels
import cv2

def get_label(img):
	lbls = [l.lower() for l in labels]

	lbls = set(lbls)

	url = "https://eastus.api.cognitive.microsoft.com/vision/v3.0/tag?language=en"

	payload = {}
	files = [
	  ('', cv2.imencode('.jpg', img)[1])
	]
	headers = {
	  'Ocp-Apim-Subscription-Key': '[API KEY]'
	}

	response = requests.request("POST", url, headers=headers, data = payload, files = files)

	res = json.loads(response.text)

	print(res)
	
	for t in res['tags']:
		if t['name'] in lbls:
			return t['name']

if __name__ == '__main__':
	get_label(cv2.imread('test.png'))