async function doStuff() {
    aniAcc();
}
async function doStuff2() {
    aniRej();
}

async function aniNext() {
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
    let button = document.getElementById("closeIcon");

    await sleep(300);
    button.style.textShadow = "0 0 10px rgb(70, 144, 255)";
    button.style.color = "rgb(170,0,0)";
    await sleep(1000);
    closeCard();
    await sleep(5000)
    openCard();
    button.style.textShadow = "";
    button.style.color = "";
}

async function aniAcc() {
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