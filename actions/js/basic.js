/**
 * sets js to sleep
 * function that uses sleep must be declared async
 * @param {Number} milliseconds
 * @see https://www.sitepoint.com/delay-sleep-pause-wait/
 */
function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

async function doStuff() {
    aniClose();
}
async function doStuff2() {
    aniPrev();
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
    let card = document.getElementById("close")

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