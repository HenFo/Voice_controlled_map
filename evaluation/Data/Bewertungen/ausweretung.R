bewertungen <- read.csv("bewertungen.csv")
durchschnitte <- read.csv("Durchschnitt.csv", sep = ";", skip = 2, col.names = c("ClassID", "Tag", "Count", "diff mean", "diff std", "prec mean", "prec std"))
class_tag <- read.csv("class_tag.csv")

bewertungen <- merge(bewertungen, class_tag, by.x = "ClassID", by.y = "Class")

par(mar=c(8,6,4,3))
boxplot(bewertungen$difficulty ~ bewertungen$Tag, xlab = "", ylab = "Bewertung: Schwierigkeit", las=2)
title("Von den Teilnehmern bewerteter Schwierigkeitsgrad")

boxplot(bewertungen$precision ~ bewertungen$Tag, xlab = "", ylab = "Bewertung: Präzision", las=2)
title("Von den Teilnehmern bewertete Präzision ihrer Kommandos")

cor(bewertungen$difficulty, bewertungen$precision)
cor(durchschnitte$diff.mean, durchschnitte$diff.std)
cor(durchschnitte$prec.mean, durchschnitte$prec.std)
