from nltk.tokenize import word_tokenize
from nltk.stem.snowball import GermanStemmer
import os
import json
import nltk
import pandas as pd

stemmer = GermanStemmer(ignore_stopwords=True)

CONFLICT_OUTPUT_PATH = os.path.join(
    "Data", "commands", "combined", "resolve_conflicts")
CREATE_VOCABULARY = os.path.join("Data", "commands", "combined", "vocabulary")


def combine_data_panning(dirpath: str, output_name: str):
    """output form:{action:{tag:[commands]}}

    Arguments:
        dirpath {str} -- [description]
        output_name {str} -- [description]
    """
    ACTION = "actions"
    TAG = "tag"
    COMMANDS = "commands"

    with open(os.path.join("Data", "commands", "stopwords.txt"), "rt") as f:
        stopwords = set(f.read().splitlines())

    document_pathes = [os.path.join(dirpath, x) for x in os.listdir(dirpath)]
    new_data = {}
    for document in document_pathes:
        with open(document, "rt") as f:
            commands = json.load(f)

        for action in commands[ACTION]:
            stemmed = [
                " ".join(stemmer.stem(x)
                         for x in word_tokenize(command) if x not in stopwords)
                for command in action[COMMANDS]
            ]

            if action[TAG] in ("0", "1", "2", "3"):
                if "0" not in new_data:
                    new_data["0"] = stemmed
                else:
                    new_data["0"].extend(stemmed)
            else:
                if action[TAG] not in new_data:
                    new_data[action[TAG]] = stemmed
                else:
                    new_data[action[TAG]].extend(stemmed)

    with open(os.path.join(CONFLICT_OUTPUT_PATH, f"{output_name}.json"), "wt") as f:
        json.dump(new_data, f)

    return new_data


def reverse_tag_commands(tag_command: dict, output_name: str):
    command_tags = {}
    for tag, commands in tag_command.items():
        for command in commands:
            if command != "":
                command = " ".join(sorted(word_tokenize(command)))
                if command in command_tags:
                    command_tags[command].append(tag)
                else:
                    command_tags[command] = [tag]

    with open(os.path.join(CONFLICT_OUTPUT_PATH, f"{output_name}.json"), "wt") as f:
        json.dump(command_tags, f)

    return command_tags


def count_tags_for_command(command_tags: dict, output_name: str):
    command_tag_count = {}
    for command, tags in command_tags.items():
        command_tag_count[command] = [
            (tag, tags.count(tag)) for tag in set(tags)]

    with open(os.path.join(CONFLICT_OUTPUT_PATH, f"{output_name}.json"), "wt") as f:
        json.dump(command_tag_count, f)

    return command_tag_count


def resolve_conflicts(command_tag_count: dict, output_name: str):
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


def calc_overall_agreement(tag_command: dict):
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


def count_commands(tag_commands: dict, output_name: str):
    count_commands = {}
    for tag, commands in tag_commands.items():
        count_commands[tag] = [(command, commands.count(command))
                               for command in set(commands)]

    with open(os.path.join(CREATE_VOCABULARY, f"{output_name}.json"), "wt") as f:
        json.dump(count_commands, f)

    return count_commands


def create_vocab(conflict_free: dict, output_name: str):
    command_count = count_commands(conflict_free, "command_count")
    vocab = {}
    for tag, commands in command_count.items():
        sorted_commands = sorted(commands, key=lambda cc: cc[1], reverse=True)

        max_border = len(conflict_free[tag])/2
        count_sum = 0
        final_assign = []
        for command, count in sorted_commands:
            if(count_sum <= max_border or count == sorted_commands[0][1]):
                final_assign.append((command, count))
                count_sum += count

        vocab[tag] = {"commands": final_assign, "total_collected_commands": len(
            conflict_free[tag]), "individual_command_count": len(set(conflict_free[tag]))}

    with open(os.path.join(CREATE_VOCABULARY, f"{output_name}.json"), "wt") as f:
        json.dump(vocab, f)

    return vocab


def transform_json_int_csv(output_path: str, json_file_path: str = None, dictionary: dict = None):
    if json_file_path != None:
        with open(json_file_path, "rt") as f:
            data = json.load(f)
    elif dictionary != None:
        data = dictionary
    else:
        assert ValueError

    c = []
    t = []
    for i, (tag, commands) in enumerate(data.items()):
        t.append([i, tag])
        for command in commands:
            c.append([command, i])

    df1 = pd.DataFrame(c, columns=["Command", "Class"])
    df1.set_index("Class", inplace=True)
    with open(output_path, "wt") as f:
        df1.to_csv(f)

    # df2 = pd.DataFrame(t, columns=["Class", "Tags"])
    # df2.set_index("Tags", inplace=True)
    # with open(output_path + "tag_class", "wt") as f:
    #     df2.to_csv(f)


if __name__ == "__main__":
    os.chdir("evaluation")

    tag_command = combine_data_panning(os.path.join(
        "Data", "commands", "collected_commands"), "combined_commands")

    agreements = [[tag, round(calc_agreement_single(commands), 2)]
                  for tag, commands in tag_command.items()]
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

    # transform_json_int_csv(os.path.join(CREATE_VOCABULARY, "conflict_free_"), dictionary=conflict_free)
    # transform_json_int_csv(os.path.join(CREATE_VOCABULARY, "final_"), dictionary=vocab)
