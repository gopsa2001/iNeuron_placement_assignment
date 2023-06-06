'''
[ NLP ]

Question - 1 :-

Take any YouTube videos link and your task is to extract the comments from
that videos and store it in a csv file and then you need define what is most
demanding topic in that videos comment section.

'''

## Answer-1 [NLP]:-


import csv
import requests

# Get the YouTube video ID
video_id = "https://www.youtube.com/watch?v=m_nQM_AkVfY"

# Get the comments from the YouTube API
url = "https://www.googleapis.com/youtube/v3/comments?part=snippet&videoId=" + video_id + "&key=YOUR_API_KEY"
response = requests.get(url)

# Parse the JSON response
comments = response.json()["items"]

# Create a CSV file to store the comments
with open("comments.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["Comment ID", "Comment", "User ID", "User Name", "Date"])
    for comment in comments:
        writer.writerow([comment["id"], comment["snippet"]["textDisplay"], comment["snippet"]["authorChannelId"], comment["snippet"]["authorDisplayName"], comment["snippet"]["publishedAt"]])

# Define the most demanding topic in the comment section
most_demanding_topic = ""
most_demanding_topic_count = 0
for comment in comments:
    topic = comment["snippet"]["textDisplay"].split(" ")[0]
    if topic not in most_demanding_topics:
        most_demanding_topics[topic] = 1
    elif topic in most_demanding_topics:
        most_demanding_topics[topic] += 1
    if most_demanding_topics[topic] > most_demanding_topic_count:
        most_demanding_topic = topic
        most_demanding_topic_count = most_demanding_topics[topic]

# Print the most demanding topic
print("The most demanding topic in the comment section is:", most_demanding_topic)



## Thank You