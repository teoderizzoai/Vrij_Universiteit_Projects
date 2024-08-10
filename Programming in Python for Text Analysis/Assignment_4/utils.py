import os
import glob
from lxml import etree

my_path = "/Users/Teo/Desktop/University/Python-for-text-analysis-master/"
path_pmb = f'{my_path}PMB/pmb-2.1.0/data/gold'


#2.a Get all token elements of a document in a given language

def get_tokens(path_to_doc):
    """
    Returns a list of token elements from "path_to_doc"
    """
    tree = etree.parse(path_to_doc)
    root = tree.getroot()
    token_elem = root.findall('xdrs/taggedtokens/tagtoken')
    return token_elem


#2.b Get token and pos from a token element
def get_token_pos(token_element):
    """
    Return a the token and part of speech (POS) tag of the token element from "token_element"
    """
    tags = token_element.findall('tags/tag')
    for tag in tags:
        if tag.get('type') == "tok":
            token = tag.text
        elif tag.get('type') == "pos":
            pos = tag.text

    return token, pos


#2.c Get document text
def get_doc_text(path_to_doc):
    """
    Return the text of the document in "path_to_doc" as a string
    """
    token_ls = []
    tree = etree.parse(path_to_doc)
    root = tree.getroot()
    tags = root.findall('xdrs/taggedtokens/tagtoken/tags/tag')
    for tag in tags:
        if tag.get('type') == "tok":
            token_ls.append(tag.text)
    join_str = " "
    return join_str.join(token_ls)



#2.d Sort documents on languages

def sort_docs(path_pmb):
    """
    Return a dictionary of every language and all their documents from "path_pmb"
    """
    lang_dict = {}
    paths = glob.glob(f"{path_pmb}/*/*/*.xml")
    for path in paths:
        basename = os.path.basename(path)
        language = basename[:2]
        try:
            lang_dict[language].add(path.rstrip(basename))
        except KeyError:
            lang_dict[language] = set([path.rstrip(basename)])
    return lang_dict