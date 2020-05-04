from nltk.tokenize import word_tokenize
from nltk.stem.snowball import GermanStemmer
import os
import json
from pprint import pprint
import nltk

stemmer = GermanStemmer(ignore_stopwords=True)


def combine_data(dirpath:str, output_name:str):
    ACTION      = "actions"
    TAG         = "tag"
    COMMANDS    = "commands"

    document_pathes = [os.path.join(dirpath, x) for x in os.listdir(dirpath)]
    new_data = {}
    for document in document_pathes:
        with open(document, "rt") as f:
            commands = json.load(f)

        for action in commands[ACTION]:
            stemmed = [
                " ".join(stemmer.stem(x) for x in word_tokenize(command))
                for command in action[COMMANDS]
            ]

            if action[TAG] not in new_data:
                new_data[action[TAG]] = stemmed
            else:
                new_data[action[TAG]].extend(stemmed)

    with open(os.path.join("Data", "commands", f"{output_name}.json"), "wt") as f:
        json.dump(new_data, f)




def normalize_command(commands:list):
    return [" ".join(stemmer.stem(word,) for word in word_tokenize(command)) for command in commands]




if __name__ == "__main__":
    train_path  = os.path.join("Data", "commands", "train")
    test_path   = os.path.join("Data", "commands", "test")
    combine_data(train_path, "combined_training_data")
    combine_data(test_path, "combined_test_data")