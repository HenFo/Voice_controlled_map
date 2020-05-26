function selection(id) {
    function hide(object) {
        object.style.visibility = "hidden";
    }

    if (id != 1) {
        hide(document.getElementById("obj1"));
    }
    if (id != 2) {
        hide(document.getElementById("obj2"));
    }
    if (id != 3) {
        hide(document.getElementById("obj3"));
    }
}