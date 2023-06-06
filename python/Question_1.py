'''
[ Python ]

Question 1: -
Write a program that takes a string as input, and counts the frequency of each word in the string, there might
be repeated characters in the string. Your task is to find the highest frequency and returns the length of the
highest-frequency word.

Note - You have to write at least 2 additional test cases in which your program will run successfully and provide
an explanation for the same.

Example input - string = “write write write all the number from from from 1 to 100”
Example output - 5

Explanation - From the given string we can note that the most frequent words are “write” and “from” and
the maximum value of both the values is “write” and its corresponding length is 5

'''


## Answer-1 [ Python]


from collections import Counter

def find_highest_frequency_word_length(input_string):
    # Split the string into words
    words = input_string.split()

    # Count the frequency of each word
    word_frequency = Counter(words)

    # Find the highest frequency
    max_frequency = max(word_frequency.values())

    # Find the length of the highest-frequency word
    highest_frequency_word_length = max(len(word) for word, freq in word_frequency.items() if freq == max_frequency)

    return highest_frequency_word_length


# Test case 1
input_str1 = "write write write all the number from from from 1 to 100"
result1 = find_highest_frequency_word_length(input_str1)
print("Length of the highest-frequency word in test case 1:", result1)


# Test case 2
input_str2 = "apple banana banana cherry cherry cherry cherry"
result2 = find_highest_frequency_word_length(input_str2)
print("Length of the highest-frequency word in test case 2:", result2)

# Test case 3
input_str3 = "this is a sample sentence with no repeated words"
result3 = find_highest_frequency_word_length(input_str3)
print("Length of the highest-frequency word in test case 3:", result3)



'''
Explanation:

Test case 1:

The input string contains repeated words such as "write" and "from".
The word "write" appears 3 times, while the word "from" appears 3 times as well.
The maximum frequency is 3, which corresponds to the word "write".
The length of the highest-frequency word "write" is 5.
Therefore, the expected output is 5.
Test case 2:

The input string contains repeated words such as "banana" and "cherry".
The word "banana" appears 2 times, while the word "cherry" appears 4 times.
The maximum frequency is 4, which corresponds to the word "cherry".
The length of the highest-frequency word "cherry" is 6.
Therefore, the expected output is 6.
Test case 3:

The input string does not have any repeated words.
Each word appears only once.
Therefore, the maximum frequency is 1, and the length of the highest-frequency word is determined by the length of any word in the input string.
In this case, we can choose any word, and the program will return its length.
Therefore, the expected output can vary depending on the specific word chosen from the input string.
'''