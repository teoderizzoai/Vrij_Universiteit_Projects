def preprocess(string):
    """
    Asks for characters and removes them from 'string' 
    """
    punctuation = []
    punctuation.append(input("What punctuation character do you want to remove?  - "))
    another = input("What other character? (if you typed them all press enter)  - ")
    while another != "":
        punctuation.append(another)
        another = input("What other character? (if you typed them all press enter)  - ")
    new_string = string
    print("")
    for char in punctuation:
        new_string = new_string.replace(char, " ")
    return new_string