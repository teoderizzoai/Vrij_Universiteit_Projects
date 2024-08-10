from utils import get_paths, get_basic_stats
import os

txt_files_dir = get_paths(input_folder="../Data/books")
book2stats = {}

for txt_file in txt_files_dir:
    print(txt_file)
    print(f"{get_basic_stats(txt_file)}\n")
    basename = os.path.basename(txt_file)
    book = basename.strip('.txt')
    book2stats[book] = get_basic_stats(txt_file)

stats2book_with_highest_value = {}
dict_sent = {}
dict_tokens = {}
dict_vocab = {}
dict_chapt = {}

for book in book2stats:
    dict_sent[book] = book2stats[book]["num_sents"]
    stats2book_with_highest_value["num_sents"] = max(dict_sent, key=dict_sent.get)

    dict_tokens[book] = book2stats[book]["num_tokens"]
    stats2book_with_highest_value["num_tokens"] = max(dict_tokens, key=dict_tokens.get)

    dict_vocab[book] = book2stats[book]["vocab_size"]
    stats2book_with_highest_value["vocab_size"] = max(dict_vocab, key=dict_vocab.get)

    dict_chapt[book] = book2stats[book]["num_chapters_or_acts"]
    stats2book_with_highest_value["num_chapters_or_acts"] = max(dict_chapt, key=dict_chapt.get)

print(f"Book with the highest:\n{stats2book_with_highest_value}")
