import requests

# url = 'http://localhost:8080/2015-03-31/functions/function/invocations'
url = 'https://9f12y6gsoe.execute-api.ap-southeast-1.amazonaws.com/production/predict'

data = {'url': 'https://drive.google.com/uc?export=view&id=1MGvOaIy94muwFCofOd88pNRszUUiwdvf'}


result = requests.post(url, json=data).json()
print(result)