import pandas as pd
import numpy as np

import os

os.chdir("evaluation")
c_dirpath = os.path.join("Data", "Bewertungen", "c")
r_dirpath = os.path.join("Data", "Bewertungen", "r")

c_docs = sorted([os.path.join(c_dirpath, x) for x in os.listdir(c_dirpath)])
r_docs = sorted([os.path.join(r_dirpath, x) for x in os.listdir(r_dirpath)])

c = []
r = []
p = []
for i, document in enumerate(c_docs):
    with open(document, "rt") as f:
        curFile = f.read().splitlines()
    
    for line in curFile:
        try:
            split = line.replace(" ", "").split(",")
            bew = list(map(lambda x: float(x), split[-2:]))
            c.append(bew)
            p.append(i)
        except:
            if split != ['']:
                c.append([-1,-1])
                p.append(i)

for document in r_docs:
    with open(document, "rt") as f:
        curFile = f.readline().replace("\n", "")
    
    seq = list(map(lambda x: int(x), curFile.split(",")))
    r.append(np.array(seq))

c = np.array(c)
r = np.array(r).reshape(-1,1)
p = np.array(p).reshape(-1,1)

cr = np.concatenate((p,r,c), 1)

with open(os.path.join("Data", "Bewertungen", "class_tag.csv"), "rt") as f:
    ct = pd.read_csv(f)

bewDF = pd.DataFrame(data=cr, columns=["ParticipantID", "ClassID", "difficulty", "precision"])
bewDF["ClassID"] = bewDF["ClassID"].astype(int)
bewDF["ParticipantID"] = bewDF["ParticipantID"].astype(int)
bewDF = bewDF[bewDF["precision"] > 0]

bewDF = bewDF.merge(ct, left_on="ClassID", right_on="Class")
bewDF = bewDF.drop("Class", 1)

bewDF = bewDF.sort_values(["ParticipantID", "ClassID"])

auswertungDF = bewDF

bewDF = bewDF.set_index(["ParticipantID", "ClassID", "Tag"])


with open(os.path.join("Data", "Bewertungen", "bewertungen.csv"), "wt") as f:
    f.write(bewDF.to_csv())

auswertungDF["count"] = auswertungDF.groupby("ClassID")["ClassID"].transform(np.size)
auswertungDF = auswertungDF.groupby(["ClassID", "Tag", "count"]).agg({"difficulty":["mean", "std"], "precision": ["mean", "std"]}).round(2)
auswertungDF = auswertungDF.sort_values("ClassID")
with open(os.path.join("Data", "Bewertungen", "Durchschnitt.csv"), "wt") as f:
    f.write(auswertungDF.to_csv())


