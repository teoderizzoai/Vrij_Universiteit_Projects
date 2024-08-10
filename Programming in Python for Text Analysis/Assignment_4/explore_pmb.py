from utils import sort_docs, get_tokens

my_path = "/Users/Teo/Desktop/University/Python-for-text-analysis-master/"
path_pmb = f'{my_path}PMB/pmb-2.1.0/data/gold'

# 3.a Printing statistics for all languages

language_doc_dict = sort_docs(path_pmb)
languages = list(language_doc_dict.keys())
token_dict = {}
for language in languages:
    num_docs = len(language_doc_dict[language])
    documents = language_doc_dict[language]
    for doc in documents:
        doc_path = f'{doc}{language}.drs.xml'
        tokens = get_tokens(doc_path)
        try:
            token_dict[language].append(tokens)
        except KeyError:
            token_dict[language] = tokens
    num_tokens = len(token_dict[language])
    print(f"{language}: num docs: {num_docs}, num tokens: {num_tokens}")


# 3.b Printing statistics for language pairs

def get_pairs(language_list):
    """
    Return a list with every possible combination with the items in "language_list"
    Also prints the amount of documents that overlap in each of these combinations
    """

    pairs = []
    for l1 in language_list:
        for l2 in language_list:
            if l1 != l2:
                if (l1, l2) not in pairs and (l2, l1) not in pairs:
                    pairs.append((l1, l2))
    for pair in pairs:
        lang1 = pair[0]
        lang2 = pair[1]
        overlap = language_doc_dict[lang1].intersection(language_doc_dict[lang2])
        num_overlap = len(overlap)
        print(f'Coverage for parallel data in {lang1} and {lang2}: {num_overlap}')
    return pairs


language_list = ['nl', 'en', 'it', 'de']
get_pairs(language_list)

# 3.c Explore parallel documents

lang1 = input("First language(nl/en/it/de): ")
lang2 = input("Second language(nl/en/it/de): ")
overlap = language_doc_dict[lang1].intersection(language_doc_dict[lang2])
for document in overlap:
    print(document)
    more = input('Would you like to continue?(yes/no)')
    if more == 'yes':
        continue
    elif more == 'no':
        break
