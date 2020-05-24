var images = [
    "images\\david-clode-e3irr6H7e5s-unsplash.jpg",
    "images\\simon-berger-qOmxP7W8svM-unsplash.jpg"
];

function next() {
    let val = document.getElementById("value");
    let image = document.getElementById("image");
    let number = parseInt(val.innerHTML);
    val.innerHTML = ++number;
    if (number < images.length && number >= 0) {
        image.src = images[number];
        image.style.visibility = "visible";
    }
    else {
        image.style.visibility = "hidden";
    }
}

function prev() {
    let val = document.getElementById("value");
    let image = document.getElementById("image");
    let number = parseInt(val.innerHTML);
    val.innerHTML = --number;
    if (number < images.length && number >= 0) {
        image.src = images[number];
        image.style.visibility = "visible";
    }
    else {
        image.style.visibility = "hidden";
    }
}


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
    aniNext();
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