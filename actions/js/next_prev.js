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