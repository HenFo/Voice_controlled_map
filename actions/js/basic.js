async function doStuff() {
    aniSelection();
}
async function doStuff2() {

}

let container = ["selection", "next-prev", "close", "accept-reject"];

async function activateContainer(containerID) {
    var visibility = "";
    for (let i = 0; i < container.length; i++) {
        let value = container[i];
        let conObj = document.getElementById(value);
        if (value == containerID) {
            visibility = conObj.style.visibility;
            conObj.style.visibility = "visible";
        } else {
            conObj.style.visibility = "hidden";
        }
    }
    if(visibility == "hidden" || visibility == "") {
        await sleep(2000);
    }
}

async function aniNext() {
    await activateContainer("next-prev");

    let button = document.getElementById("next");
    await sleep(300);
    button.style.backgroundColor = "#d1d1d1";
    button.style.boxShadow = "0 0 5px rgb(70, 144, 255)";
    await sleep(700);
    button.style.backgroundColor = "";
    button.style.boxShadow = "";

    next();
}

async function aniPrev() {
    await activateContainer("next-prev");

    let button = document.getElementById("prev");
    await sleep(300);
    button.style.backgroundColor = "#d1d1d1";
    button.style.boxShadow = "0 0 5px rgb(70, 144, 255)";
    await sleep(700);
    button.style.backgroundColor = "";
    button.style.boxShadow = "";
    prev();
}

async function aniClose() {
    await activateContainer("close");

    let button = document.getElementById("closeIcon");

    await sleep(300);
    button.style.textShadow = "0 0 10px rgb(70, 144, 255)";
    button.style.color = "rgb(170,0,0)";
    await sleep(1500);
    closeCard();
    await sleep(5000)
    openCard();
    button.style.textShadow = "";
    button.style.color = "";
}

async function aniAcc() {
    await activateContainer("accept-reject");

    let button = document.getElementById("acc");
    await sleep(300);
    button.style.boxShadow = "0 0 10px rgb(70, 144, 255)";
    button.style.backgroundColor = "rgb(0, 160, 0)";
    await sleep(1500);
    button.style.backgroundColor = "";
    button.style.boxShadow = "";
    accept();
    await sleep(5000)
    let popup = document.getElementById("positiv");
    popup.style.top = "";
    popup.style.visibility = "";
}
async function aniRej() {
    await activateContainer("accept-reject");

    let button = document.getElementById("rej");
    await sleep(300);
    button.style.boxShadow = "0 0 10px rgb(70, 144, 255)";
    button.style.backgroundColor = "rgb(255, 59, 59)";
    await sleep(1500);
    button.style.backgroundColor = "";
    button.style.boxShadow = "";
    reject();
    await sleep(5000)
    let popup = document.getElementById("negativ");
    popup.style.top = "";
    popup.style.visibility = "";
}

async function aniSelection() {
    await activateContainer("selection");

    let id = Math.round(Math.random() * 2) + 1;
    let obj = document.getElementById("obj" + id);
    await sleep(300);
    obj.style.boxShadow = "0 0 20px rgb(70, 144, 255)";
    obj.style.backgroundColor = "rgb(170, 170, 170)";
    await sleep(1500);
    obj.style.boxShadow = "";
    obj.style.backgroundColor = "";
    selection(id);
    await sleep(5000);
    for (let i = 1; i <= 3; i++) {
        document.getElementById("obj" + i).style.visibility = "";
    }

}