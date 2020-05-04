var intents = {
    "actions": []
};

var currentClass = {
    "tag": null,
    "commands": []
};

function save() {
    let hClass = document.getElementById("class");
    let hText = document.getElementById("command");

    if (hClass.value != "" && hText.value != "") {

        if (!hClass.disabled && hClass.value != currentClass.tag) {
            currentClass.tag = hClass.value;
            hClass.disabled = true;
        }

        currentClass.commands.push(hText.value);
        console.log(currentClass);
        hText.value = "";
    } else {
        alert("Bitte Text eingeben");
    }

}

function finish() {
    if (currentClass.tag != null && currentClass.commands.length > 0) {
        if (confirm("Klasse speichern?")) {
            intents.actions.push(currentClass);
            let docClass = document.getElementById("class");
            docClass.disabled = false;
            docClass.value = "";

            currentClass = {
                "tag": null,
                "commands": []
            };

            console.log(intents);
            document.getElementById("endSession").disabled = false;
        }
    } else {
        alert("keine Klasse gespeichert");
    }

}

function endSession() {
    if (confirm("Session wirklich beenden?")) {
        var blob = new Blob([JSON.stringify(intents)], {
            type: "text/plain;charset=iso-8859-1"
        });

        intents = {
            "actions": []
        };
        currentClass = {
            "tag": null,
            "commands": []
        };
        document.getElementById("endSession").disabled = true;
        let docClass = document.getElementById("class");
        docClass.disabled = false;
        docClass.value = "";
        document.getElementById("command").value = "";
        saveAs(blob, "intents.txt");
    }
}