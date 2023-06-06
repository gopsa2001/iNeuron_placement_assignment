'''
[Python]

Question 4 -

Write a program to download the data from the link given below and then read the data and convert the into
the proper structure and return it as a CSV file.

Link - https://data.nasa.gov/resource/y77d-th95.json

Note - Write code comments wherever needed for code understanding.

'''


## Answer-4 [Python] :-

import requests
import pandas as pd

# Downloading the data from the provided link
url = "https://data.nasa.gov/resource/y77d-th95.json"
response = requests.get(url)

# Converting the data into a pandas dataframe
df = pd.read_json(response.content)

# Saving the dataframe to a CSV file
df.to_csv("nasa_data.csv", index=False)

print("Data saved to nasa_data.csv")


'''
This code downloads the data from the provided link using the requests library.
It then converts the data into a pandas dataframe using the read_json() function.
Finally, it saves the dataframe to a CSV file using the to_csv() function.

'''