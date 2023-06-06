'''
[Python]

Question 5 -

Write a program to download the data from the given API link and then extract the following data with
proper formatting

Link - http://api.tvmaze.com/singlesearch/shows?q=westworld&embed=episodes

Note - Write proper code comments wherever needed for the code understanding
'''


## Answer-5 [Python] :-

import requests

# Downloading the data from the provided link
url = "http://api.tvmaze.com/singlesearch/shows?q=westworld&embed=episodes"
response = requests.get(url)

# Extracting the required data
data = response.json()
show_name = data["name"]
show_type = data["type"]
show_language = data["language"]
show_genres = ", ".join(data["genres"])
show_status = data["status"]
show_runtime = data["runtime"]
show_summary = data["summary"].replace("<p>", "").replace("</p>", "").replace("<b>", "").replace("</b>", "")

print("Show Name:", show_name)
print("Show Type:", show_type)
print("Show Language:", show_language)
print("Show Genres:", show_genres)
print("Show Status:", show_status)
print("Show Runtime:", show_runtime)
print("Show Summary:", show_summary)


'''
Explanation :- This code downloads the data from the provided link using the requests library.
It then extracts the required data from the JSON response and formats it as required.

'''