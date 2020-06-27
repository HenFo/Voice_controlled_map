from nltk.tokenize import word_tokenize
from nltk.stem.snowball import GermanStemmer
import os
import json
import nltk
import pandas as pd

stemmer = GermanStemmer(ignore_stopwords=True)

CONFLICT_OUTPUT_PATH = os.path.join("Data", "commands", "combined", "resolve_conflicts")
CREATE_VOCABULARY = os.path.join("Data", "commands", "combined", "vocabulary")

def combine_data(dirpath:str, output_name:str):
    """output form:{tag:[commands]}

    Arguments:
        dirpath {str} -- [description]
        output_name {str} -- [description]
    """
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

    with open(os.path.join(CONFLICT_OUTPUT_PATH, f"{output_name}.json"), "wt") as f:
        json.dump(new_data, f)
    
    return new_data


def reverse_tag_commands(tag_command:dict, output_name:str):
    command_tags = {}
    for tag, commands in tag_command.items():
        for command in commands:
            if command in command_tags:
                command_tags[command].append(tag)
            else:
                command_tags[command] = [tag]
    
    with open(os.path.join(CONFLICT_OUTPUT_PATH, f"{output_name}.json"), "wt") as f:
        json.dump(command_tags, f)

    return command_tags


def count_tags_for_command(command_tags:dict, output_name:str):
    command_tag_count = {}
    for command, tags in command_tags.items():
        command_tag_count[command] = [(tag,tags.count(tag)) for tag in set(tags)]
    
    with open(os.path.join(CONFLICT_OUTPUT_PATH, f"{output_name}.json"), "wt") as f:
        json.dump(command_tag_count, f)
    
    return command_tag_count

def resolve_conflicts(command_tag_count:dict, output_name:str):
    conflict_free_tag_commands = {}
    for command, tag_count in command_tag_count.items():
        tag, count = max(tag_count, key=lambda tc: tc[1])

        if tag in conflict_free_tag_commands:
            conflict_free_tag_commands[tag].extend([command] * count)
        else:
            conflict_free_tag_commands[tag] = [command] * count
    
    with open(os.path.join(CONFLICT_OUTPUT_PATH, f"{output_name}.json"), "wt") as f:
        json.dump(conflict_free_tag_commands, f)

    return conflict_free_tag_commands

def calc_overall_agreement(tag_command:dict):
    top = 0
    for commands in tag_command.values():
        top += calc_agreement_single(commands)

    return top/len(tag_command)

def calc_agreement_single(commands):
    command_count = [commands.count(command) for command in set(commands)]
    Ar = 0
    for i in command_count:
        Ar += (i/len(commands))**2
    
    return Ar


def count_commands(tag_commands:dict, output_name:str):
    count_commands = {}
    for tag, commands in tag_commands.items():
        count_commands[tag] = [(command, commands.count(command)) for command in set(commands)]
    
    with open(os.path.join(CREATE_VOCABULARY, f"{output_name}.json"), "wt") as f:
        json.dump(count_commands, f)

    return count_commands
            

def create_vocab(conflict_free:dict, output_name:str):
    command_count = count_commands(conflict_free, "command_count")
    vocab = {}
    for tag, commands in command_count.items():
        sorted_commands = sorted(commands, key=lambda cc:cc[1], reverse=True)

        max_border = len(conflict_free[tag])/2
        count_sum = 0
        final_assign = []
        for command, count in sorted_commands:
            if(count_sum > max_border and not count == sorted_commands[0][1]):
                break

            final_assign.append(command)
            count_sum += count

        
        vocab[tag] = final_assign
    
    with open(os.path.join(CREATE_VOCABULARY, f"{output_name}.json"), "wt") as f:
        json.dump(vocab, f)

    return vocab
    

        
tag_command = combine_data(os.path.join("Data", "commands", "collected_commands"), "combined_commands")

agreements = [[tag, round(calc_agreement_single(commands), 2)] for tag, commands in tag_command.items()]
agreements.append(["A", round(calc_overall_agreement(tag_command), 2)])

df = pd.DataFrame(agreements, columns=["Tag", "Agreement"])
df.set_index("Tag", inplace=True)

with open(os.path.join("Data", "commands", "agreement.csv"), "wt") as f:
        df.to_csv(f)
        

# resolve conflicts
command_tags = reverse_tag_commands(tag_command, "reversed_command_tags")
command_tag_count = count_tags_for_command(command_tags, "tag_count")
conflict_free = resolve_conflicts(command_tag_count, "conflict_free")

# create vocab
vocab = create_vocab(conflict_free, "final_vocabulary")