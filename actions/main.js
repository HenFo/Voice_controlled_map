mapboxgl.accessToken = 'pk.eyJ1IjoiZGltYm9kdW1ibyIsImEiOiJjamplN2t4dXYxaDY2M2twOTQzMXNocjc2In0.g9BJj267dR8RBxBBgi2fyQ';
var map = new mapboxgl.Map({
    container: 'map', // container id
    style: 'mapbox://styles/mapbox/streets-v11',
    center: [-74.5, 40], // starting position
    zoom: 9 // starting zoom
});

var compass = new mapboxgl.NavigationControl({
    showZoom: false,
    visualizePitch: false
});
map.addControl(compass, 'top-right');



const ANIMATION_DURATION = 2000;


function doStuff() {
    showResults();
}
function doStuff2() {
    hideResults();
}

/*
MENU NAVIGATION
*/

function hideMenu() {
    $("#menu").css("left", "-15.5em");
}

function showMenu() {
    $("#menu").css("left", ".75em");
}
function hideResults() {
    $("#results").css("left", "-15.5em");
}

function showResults() {
    $("#results").css("left", ".75em");
}

function showSearch() {
    $("#searchbar").css("top", "1em");
}
function hideSearch() {
    $("#searchbar").css("top", "-5em");

}



/*
MAP NAVIGATION
*/

function rotate(bearing) {
    map.rotateTo(bearing);
}

function pitch(pitch) {
    map.easeTo({pitch: pitch});
}

function resetView() {
    map.resetNorthPitch({duration: ANIMATION_DURATION});
}

function up() {
    let bounds = map.getBounds();
    let newCenter = new mapboxgl.LngLat(bounds.getCenter().lng, bounds.getNorth());

    console.log(bounds);
    console.log(newCenter);
    
    

    map.panTo(newCenter, {duration: ANIMATION_DURATION});
}

function down() {
    let bounds = map.getBounds();
    let newCenter = new mapboxgl.LngLat(bounds.getCenter().lng, bounds.getSouth());

    map.panTo(newCenter, {duration: ANIMATION_DURATION});
}

function left() {
    let bounds = map.getBounds();
    let newCenter = new mapboxgl.LngLat(bounds.getWest(), bounds.getCenter().lat);

    map.panTo(newCenter, {duration: ANIMATION_DURATION});
}

function right() {
    let bounds = map.getBounds();
    let newCenter = new mapboxgl.LngLat(bounds.getEast(), bounds.getCenter().lat);

    map.panTo(newCenter, {duration: ANIMATION_DURATION});
}

function upRight() {
    let bounds = map.getBounds();
    map.panTo(bounds.getNorthEast(), {duration: ANIMATION_DURATION})
}

function upLeft() {
    let bounds = map.getBounds();
    map.panTo(bounds.getNorthWest(), {duration: ANIMATION_DURATION})
}

function downRight() {
    let bounds = map.getBounds();
    map.panTo(bounds.getSouthEast(), {duration: ANIMATION_DURATION})
}

function downLeft() {
    let bounds = map.getBounds();
    map.panTo(bounds.getSouthWest(), {duration: ANIMATION_DURATION})
}

function zoomIn() {
    map.zoomIn({duration: ANIMATION_DURATION})
}

function zoomOut() {
    map.zoomOut({duration: ANIMATION_DURATION})
}