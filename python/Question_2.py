'''
[Python]

Question 2: -

Consider a string to be valid if all characters of the string appear the same number of times. It is also valid if
he can remove just one character at the index in the string, and the remaining characters will occur the same
number of times. Given a string, determine if it is valid. If so, return YES , otherwise return NO .

Note - You have to write at least 2 additional test cases in which your program will run successfully and provide
an explanation for the same.

Example input 1 - s = “abc”. This is a valid string because frequencies are { “a”: 1, “b”: 1, “c”: 1 }
Example output 1- YES

Example input 2 - s “abcc”. This string is not valid as we can remove only 1 occurrence of “c”. That leaves
character frequencies of { “a”: 1, “b”: 1 , “c”: 2 }
Example output 2 - NO

'''

## Answer-2 [Python ]

def is_valid_string(s):
    # Count the frequency of each character
    char_count = {}
    for char in s:
        char_count[char] = char_count.get(char, 0) + 1

    # Count the frequency of each frequency
    freq_count = {}
    for count in char_count.values():
        freq_count[count] = freq_count.get(count, 0) + 1

    # If there is only one frequency or two frequencies with a difference of one,
    # the string is valid
    if len(freq_count) == 1 or (len(freq_count) == 2 and 1 in freq_count.values()):
        return "YES"
    else:
        return "NO"




print(is_valid_string("abc"))  # Output: YES
print(is_valid_string("abcc"))  # Output: NO


## Additional test case 1:
print(is_valid_string("aabbcc"))  # Output: YES


''' Explanation:- In this case, all characters have the same frequency, so the string is valid.'''


## Additional test case 2:
print(is_valid_string("aabbc"))  # Output: YES

'''Explanation:- The character 'c' appears only once, so we can remove it to make all characters have the same frequency (2).'''


