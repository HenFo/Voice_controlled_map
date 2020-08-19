from nltk.tokenize import word_tokenize
from nltk.stem.snowball import GermanStemmer
import os
import json
import nltk
import pandas as pd
import numpy as np

stemmer = GermanStemmer(ignore_stopwords=True)

CONFLICT_OUTPUT_PATH = os.path.join("Output")
CREATE_VOCABULARY = os.path.join("Output")


def combine_data_panning(dirpath: str, output_name: str = None):
    ACTION = "actions"
    TAG = "tag"
    COMMANDS = "commands"

    # with open(os.path.join("Data", "stopwords.txt"), "rt") as f:
    #     stopwords = set(f.read().splitlines())

    document_pathes = [os.path.join(dirpath, x) for x in os.listdir(dirpath)]
    new_data = {}
    for i, document in enumerate(document_pathes):

        with open(document, "rt") as f:
            commands = json.load(f)

        repeat = set()
        for action in commands[ACTION]:
            tag = action[TAG] if action[TAG] not in (
                "0", "1", "2", "3") else "0"

            if tag not in new_data:
                new_data[tag] = []

            normalized = []

            for j, command in enumerate(action[COMMANDS]):
                c = " ".join(stemmer.stem(x)
                             for x in sorted(word_tokenize(command)))

                if c not in repeat:
                    normalized.append({"c": c, "penalty": j, "proband": i+1})
                    if tag == "0":
                        repeat.add(c)

            new_data[tag].extend(normalized)

    if output_name:
        with open(os.path.join(CONFLICT_OUTPUT_PATH, f"{output_name}.json"), "wt") as f:
            json.dump(new_data, f)

    return new_data

def combine_data_first(dirpath: str, output_name: str = None):
    ACTION = "actions"
    TAG = "tag"
    COMMANDS = "commands"

    # with open(os.path.join("Data", "stopwords.txt"), "rt") as f:
    #     stopwords = set(f.read().splitlines())

    document_pathes = [os.path.join(dirpath, x) for x in os.listdir(dirpath)]
    new_data = {}
    for i, document in enumerate(document_pathes):

        with open(document, "rt") as f:
            commands = json.load(f)

        repeat = set()
        for action in commands[ACTION]:
            tag = action[TAG] if action[TAG] not in (
                "0", "1", "2", "3") else "0"

            if tag not in new_data:
                new_data[tag] = []

            normalized = []

            j = 0
            try:
                command = action[COMMANDS][0]
                c = " ".join(stemmer.stem(x)
                                for x in sorted(word_tokenize(command)))

                if c not in repeat:
                    normalized.append({"c": c, "penalty": j, "proband": i+1})
                    if tag == "0":
                        repeat.add(c)

                new_data[tag].extend(normalized)
            except IndexError:
                continue

    if output_name:
        with open(os.path.join(CONFLICT_OUTPUT_PATH, f"{output_name}.json"), "wt") as f:
            json.dump(new_data, f)

    return new_data


def combine_data_allDir(dirpath: str, output_name: str = None):
    ACTION = "actions"
    TAG = "tag"
    COMMANDS = "commands"

    # with open(os.path.join("Data", "stopwords.txt"), "rt") as f:
    #     stopwords = set(f.read().splitlines())

    document_pathes = [os.path.join(dirpath, x) for x in os.listdir(dirpath)]
    new_data = {}
    for i, document in enumerate(document_pathes):

        with open(document, "rt") as f:
            commands = json.load(f)

        for action in commands[ACTION]:
            tag = action[TAG]
            if tag not in new_data:
                new_data[tag] = []

            normalized = []

            for j, command in enumerate(action[COMMANDS]):
                c = " ".join(stemmer.stem(x)
                             for x in sorted(word_tokenize(command)))

                normalized.append({"c": c, "penalty": j, "proband": i+1})

            new_data[tag].extend(normalized)

    if output_name:
        with open(os.path.join(CONFLICT_OUTPUT_PATH, f"{output_name}.json"), "wt") as f:
            json.dump(new_data, f)

    return new_data

def combine_data_allDir_first(dirpath: str, output_name: str = None):
    ACTION = "actions"
    TAG = "tag"
    COMMANDS = "commands"

    # with open(os.path.join("Data", "stopwords.txt"), "rt") as f:
    #     stopwords = set(f.read().splitlines())

    document_pathes = [os.path.join(dirpath, x) for x in os.listdir(dirpath)]
    new_data = {}
    for i, document in enumerate(document_pathes):

        with open(document, "rt") as f:
            commands = json.load(f)

        for action in commands[ACTION]:
            tag = action[TAG]
            if tag not in new_data:
                new_data[tag] = []

            normalized = []

            try:
                j = 0,
                command = action[COMMANDS][0]
                c = " ".join(stemmer.stem(x)
                                for x in sorted(word_tokenize(command)))

                normalized.append({"c": c, "penalty": j, "proband": i+1})

                new_data[tag].extend(normalized)
            except IndexError:
                continue

    if output_name:
        with open(os.path.join(CONFLICT_OUTPUT_PATH, f"{output_name}.json"), "wt") as f:
            json.dump(new_data, f)

    return new_data


def remove_duplicates(tag_command: dict, output_name: str = None):
    command_tags = {}
    for tag, commands in tag_command.items():
        for command in commands:
            if command["c"] not in command_tags:
                command_tags[command["c"]] = []
            command_tags[command["c"]].append(
                {"tag": tag, "penalty": command["penalty"], "proband": command["proband"]})

    clean_tag_command = {}
    for command, metas in command_tags.items():

        tags = [t["tag"] for t in metas]
        max_tag = max(tags, key=lambda x: tags.count(x))

        if max_tag not in clean_tag_command:
            clean_tag_command[max_tag] = []

        clean_tag_command[max_tag].extend(
            [{"c": command, "penalty": m["penalty"], "proband": m["proband"]} for m in metas])

    if output_name:
        with open(os.path.join(CONFLICT_OUTPUT_PATH, f"{output_name}.json"), "wt") as f:
            json.dump(clean_tag_command, f)

    return clean_tag_command


def calc_overall_agreement(tag_command: dict):
    top = 0
    for commands in tag_command.values():
        top += calc_agreement_single(commands)

    return top/len(tag_command)


def calc_agreement_single(commands):
    command_count = [commands.count(command) for command in set(commands)]
    P_len = len(commands)
    Ar = 0
    for i in command_count:
        Ar += (i/P_len)**2

    return (P_len/(P_len - 1)) * Ar - (1/(P_len - 1))


def count_commands(tag_commands: dict, output_name: str = None):
    count_commands = {}
    for tag, commands in tag_commands.items():
        commands = [c["c"] for c in commands]
        count_commands[tag] = [(command, commands.count(command))
                               for command in set(commands)]

    if output_name:
        with open(os.path.join(CREATE_VOCABULARY, f"{output_name}.json"), "wt") as f:
            json.dump(count_commands, f)

    return count_commands


def create_vocab(conflict_free: dict, output_name: str = None):
    command_count = count_commands(conflict_free)
    vocab = {}
    for tag, commands in command_count.items():
        sorted_commands = sorted(commands, key=lambda cc: cc[1], reverse=True)

        max_border = len(conflict_free[tag])/2
        count_sum = 0
        final_assign = []
        for command, count in sorted_commands:
            if count_sum <= max_border:
                final_assign.append((command, count))
                count_sum += count

        vocab[tag] = {"commands": final_assign, "total_collected_commands": len(
            conflict_free[tag]), "individual_command_count": len(command_count[tag])}

    if output_name:
        with open(os.path.join(CREATE_VOCABULARY, f"{output_name}.json"), "wt") as f:
            json.dump(vocab, f)

    return vocab


def einigungsrate(combined: dict, num_teilnehmer: int, output_name: str = None):
    command_count = count_commands(combined)
    agreement = []
    for tag, commands in command_count.items():
        agreement.append(
            [tag, (num_teilnehmer/(num_teilnehmer-1)) * (max(commands, key=lambda cc: (cc[1]/num_teilnehmer))[1] / num_teilnehmer) - (1/(num_teilnehmer-1))])

    df = pd.DataFrame(agreement, columns=["Tag", "Einigung"]).set_index("Tag")
    if output_name:
        with open(os.path.join(CREATE_VOCABULARY, f"{output_name}.csv"), "wt") as f:
            df.to_csv(f)

    return agreement


def einigungsrate_penalty(combined: dict, num_teilnehmer: int, decreasing_rate: float = 0.5, output_name: str = None):
    dfs = []
    for tag, metas in combined.items():
        commands = []
        penaltyes = []
        for meta in metas:
            commands.append(meta["c"])
            penaltyes.append(meta["penalty"])

        df = pd.DataFrame({"t": tag, "c": commands, "p": penaltyes})
        df = df.groupby(["t", "c"]).agg(
            {"p": ["mean", "count"]}).reset_index(level=["t", "c"])
        dfs.append(df)

    agreements = []
    for df in dfs:
        tag = df["t"][0]
        tag_agreement = []
        for i, row in df.iterrows():
            val = ((num_teilnehmer / (num_teilnehmer - 1)) * (row["p", "count"] / num_teilnehmer) - (1 / (num_teilnehmer - 1))) * \
                (2/(1+2**(decreasing_rate * row["p", "mean"])))
            tag_agreement.append(val)

        agreements.append([tag, df["c"].iloc[tag_agreement.index(
            max(tag_agreement))], max(tag_agreement)])

    if output_name:
        with open(os.path.join(CREATE_VOCABULARY, f"{output_name}.csv"), "wt") as f:
            pd.DataFrame(agreements, columns=["Tag", "Command", "Einigung"]).set_index(
                "Tag").to_csv(f)

    return agreements


def guessability(vocab: dict, output_name: str = None):
    guess = []
    for tag, meta in vocab.items():
        commands = meta["commands"]
        summe = sum([x[1] for x in commands])
        guess.append([tag, summe/meta["total_collected_commands"]])

    if output_name:
        with open(os.path.join("Data", "class_tag.csv"), "rt") as f:
            ct = pd.read_csv(f).set_index("Class")

        with open(os.path.join(CREATE_VOCABULARY, f"{output_name}.csv"), "wt") as f:
            pd.DataFrame(guess, columns=["Tag", "Guessability"]) \
                .set_index("Tag") \
                .merge(ct, left_on="Tag", right_on="Class") \
                .to_csv(f)

    return guess


def calc_overall_agreement(tag_command: dict):
    top = 0
    for commands in tag_command.values():
        top += calc_agreement_single(commands)

    return top/len(tag_command)


def calc_agreement_single(commands):
    commands = [c["c"] for c in commands]
    command_count = [commands.count(command) for command in set(commands)]
    P_len = len(commands)
    Ar = 0
    for i in command_count:
        Ar += (i/P_len)**2

    return (P_len/(P_len - 1)) * Ar - (1/(P_len - 1))


if __name__ == "__main__":
    os.chdir("evaluation")

    path = os.path.join("Data", "Bewegungen alle")

    combined = combine_data_allDir_first(path, "combined_all_f")
    # resolve conflicts
    conflict_free = remove_duplicates(combined, "conflict_free_all_f")

    # # # create vocab
    vocab = create_vocab(conflict_free, "final_vocabulary_all_f")

    # with open(os.path.join("Data", "class_tag.csv"), "rt") as f:
    #     ct = pd.read_csv(f).set_index("Class")

    # top_vocab = [[t, c["commands"][0][0], c["commands"][0][1]] for t,c in vocab.items()]
    # top_vocab = pd.DataFrame(top_vocab, columns=["Tag", "Kommando", "#Nennungen"]).set_index("Tag")
    # top_vocab = top_vocab.merge(ct, left_index=True, right_index=True).set_index("Tag")

    # with open(os.path.join("Output", "top_vocab.csv"), "wt") as f:
    #     top_vocab.to_csv(f)

    # # Einigungsrate
    # e = einigungsrate(combined, 19, output_name="e")
    # e = einigungsrate_penalty(combined, 19, output_name="e_p")
    # g = guessability(vocab, "guessability")

    # Agreement / Uebereinstimmungsrate
    # a = [[t, calc_agreement_single(c)] for t,c in combined.items()]
    # a.append(["A", calc_overall_agreement(combined)])
    # df = pd.DataFrame(a, columns=["Tag", "Agreement"]).set_index("Tag")
    
    # df = df.merge(ct, left_index=True, right_index=True, how="left").set_index("Tag")
    
    # with open(os.path.join("Output", "agreement.csv"), "wt") as f:
    #     df.to_csv(f)