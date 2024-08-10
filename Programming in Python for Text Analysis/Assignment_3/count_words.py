from utils_3a import preprocess
def count(string):
    """
    After calling 'preprocess()', prints a dictionary containing 
    the amount of time each word appeared in 'string'
    """
    words = preprocess(string).split()
    word_dictionary = {}
    for word in words:
        if word in word_dictionary:
            word_dictionary[word] +=1
        else:
            word_dictionary[word] = 1
    print(word_dictionary)

count('this is a (tricky) test test test')
