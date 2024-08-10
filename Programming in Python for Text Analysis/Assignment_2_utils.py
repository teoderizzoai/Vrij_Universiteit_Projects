#function for assignment 2, exercise 6d
def clean_text_general(text, chars_to_remove={'\n', ',', '.', '"'}):
    new_text = text
    for char in chars_to_remove:
        new_text = new_text.replace(char, " ")
    return str(new_text)