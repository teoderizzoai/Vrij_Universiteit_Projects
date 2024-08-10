import os
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize


def get_paths(input_folder):
    """
    retuns a list of all the paths to .txt file inside the input folder 
    """
    ###taken from [https://stackoverflow.com/questions/9623398/text-files-in-a-dir-and-store-the-file-names-in-a-list-python][29/09/2021]

    list_txt=[]
    dir_listing = os.listdir(input_folder)
    for item in dir_listing:
        if ".txt" in item:
            list_txt.append(input_folder+'/'+item)
    return list_txt

#get_paths(input_folder="../Data/books")

def get_basic_stats(txt_path):
    """
    Takes a .txt path and return a dictionary containing the following information:
    number of sentences, number of tokens, vocabolary size and number of chapters (or acts)

    The 30 most present tokens in the .txt file are printed in in a new .txt file, one per each line
    """
    infile = open(txt_path, encoding='utf-8')
    text = infile.read()
    
    unique_words = set()
    sent_count = 0
    word_count = 0
    chapter_count = 0
    token_frequency = {}
    for sent in sent_tokenize(text):
        sent_count +=1
    for word in word_tokenize(text):
        word_count +=1
        unique_words.add(word)
        if word=="CHAPTER" or word=="Chapter" or word=="ACT":
            chapter_count +=1
        if word in token_frequency:
            token_frequency[word] +=1
        else:
            token_frequency[word] = 1

    ###taken from [https://www.tutorialspoint.com/how-to-sort-a-dictionary-in-python][30/09/2021]
    sorted_token_frequency = sorted(token_frequency.items(), key=lambda x:x[1], reverse=True)
    top_30_tokens = []
    for token, frequency in sorted_token_frequency:
        if len(top_30_tokens) < 30:
           top_30_tokens.append(token)


    dictionary = {}
    dictionary["num_sents"]= sent_count
    dictionary["num_tokens"]= word_count
    dictionary["vocab_size"]= len(unique_words)
    dictionary["num_chapters_or_acts"]=chapter_count
    dictionary["top_30_tokens"]= top_30_tokens

    new_file =open("top_30_"+os.path.basename(txt_path), "w", encoding= "utf-8")
    for token in top_30_tokens:
        new_file.write(f"{token}\n")


    return dictionary