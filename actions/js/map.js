mapboxgl.accessToken = 'pk.eyJ1IjoiZGltYm9kdW1ibyIsImEiOiJjamplN2t4dXYxaDY2M2twOTQzMXNocjc2In0.g9BJj267dR8RBxBBgi2fyQ';
var map = new mapboxgl.Map({
    container: 'map',
    style: 'mapbox://styles/mapbox/streets-v11',
    center: [-74.5, 40],
    zoom: 9,
    // interactive: false
});

var compass = new mapboxgl.NavigationControl({
    showZoom: false,
    visualizePitch: false
});
map.addControl(compass, 'top-right');

var distanceContainer = document.getElementById('distance');

// https://stackoverflow.com/questions/4467539/javascript-modulo-gives-a-negative-result-for-negative-numbers
Number.prototype.mod = function(n) {
    return ((this%n)+n)%n;
};



const ANIMATION_DURATION = 2000;

/**
 * sets js to sleep
 * function that uses sleep must be declared async
 * @param {Number} milliseconds
 * @see https://www.sitepoint.com/delay-sleep-pause-wait/
 */
function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}


function resetPosition() {
    map.jumpTo({
        center: [-74.5, 40],
        zoom: 9,
        pitch: 0,
        bearing: 0
    });
}


async function doStuff() {
    next();
}

function doStuff2() {
    sateliteBasemap();
}

function doStuff3() {
    setPoint();
}

/*
BASIC INTERACTIONS
*/

function accept() {    
    let popup = document.getElementById("positiv");
    popup.style.top = "50%";
    popup.style.visibility = "visible";
}

function reject() {
    let popup = document.getElementById("negativ");
    popup.style.top = "50%";
    popup.style.visibility = "visible";
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
}


let marker = [];
let position = 0;
function initPrevNext() {
    // Empire State, WiBRidge, Liberty
    let waypoints = [
        [-73.985632, 40.748473],
        [-73.972016, 40.713601],
        [-74.044539, 40.689398]
    ];
    
    for (let i = 0; i < waypoints.length; i++) {
        marker.push(new mapboxgl.Marker()
            .setLngLat(waypoints[i])
            .addTo(map));
    }

    map.jumpTo({
        center: waypoints[0],
        zoom: 12
    });

    showResults();
    let obj = document.querySelector("#resultList > li:nth-child(1)");
    obj.style.fontWeight = "bolder";
    obj.style.color = "#525252";
}

async function next() {
    const button = document.getElementById("next");
    button.style.backgroundColor = "#c2c2c2";
    await sleep(2000);
    button.style.backgroundColor= "";

    let obj = document.querySelector(`#resultList > li:nth-child(${position+1})`);
    obj.style.fontWeight = "";
    obj.style.color = "";

    position = (position +1) % 3;

    obj = document.querySelector(`#resultList > li:nth-child(${position+1})`);
    obj.style.fontWeight = "bolder";
    obj.style.color = "#525252";


    map.flyTo({center: marker[position].getLngLat(), zoom:12, speed:0.3});
}
async function prev() {
    const button = document.getElementById("prev");
    button.style.backgroundColor = "#c2c2c2";
    await sleep(2000);
    button.style.backgroundColor= "";

    let obj = document.querySelector(`#resultList > li:nth-child(${position+1})`);
    obj.style.fontWeight = "";
    obj.style.color = "";

    position = position -1  < 0 ? 2 : position -1;

    obj = document.querySelector(`#resultList > li:nth-child(${position+1})`);
    obj.style.fontWeight = "bolder";
    obj.style.color = "#525252";


    map.flyTo({center: marker[position].getLngLat(), zoom:12, speed:0.3});
}

function stopNextPrev() {
    for (let i = 0; i < marker.length; i++) {
        marker[i].remove();        
    }
    marker = [];
    position = 0;
    hideResults();
    resetPosition();
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

function hideHelp() {
    $("#help").css("left", "-15.5em");
}

function showHelp() {
    $("#help").css("left", ".75em");
}

function showSearch() {
    $("#searchbar").css("top", "1em");
}

function hideSearch() {
    $("#searchbar").css("top", "-5em");

}

// function selectResult(number) {
//     clearSelection();
//     $(`#resultList li:nth-child(${number})`).css("background-color", "gray");
// }

// function clearSelection() {
//     $("#resultList").children().css("background-color", "unset");
// }

/*
MAP MANIPULATION
*/

function sateliteBasemap() {
    map.setStyle('mapbox://styles/mapbox/satellite-v9');
}

function vectorBasemap() {
    map.setStyle('mapbox://styles/mapbox/streets-v11');
}

async function fakeUserLocation() {
    let center = map.getCenter();
    let zoom = map.getZoom();
    let coordinates = [-75, 40];
    let point = {
        "type": "Point",
        "coordinates": coordinates
    };
    map.addSource('point', {
        'type': 'geojson',
        'data': point
    });


    map.addLayer({
        'id': 'circle',
        'source': 'point',
        'type': 'circle',
        'paint': {
            'circle-radius': 10,
            'circle-color': '#0070bf'
        }
    });

    var popup = new mapboxgl.Popup({
            closeOnClick: false
        })
        .setLngLat(coordinates)
        .setHTML('<p>Hier bin ich<p>')
        .addTo(map);

    map.flyTo({
        center: coordinates,
        zoom: 14
    })

    await sleep(7000);
    map.removeLayer("circle");
    map.removeSource("point");
    popup.remove();
    map.jumpTo({
        center: center,
        zoom: zoom
    });

}


var markers = [];

function addMarker() {
    let center = map.getCenter();
    var marker = new mapboxgl.Marker()
        .setLngLat(center)
        .addTo(map);
    markers.push(marker);
}

function clearMarker() {
    for (let i = 0; i < markers.length; i++) {
        markers[i].remove();
    }
    markers = [];
}


var geojson = {
    'type': 'FeatureCollection',
    'features': []
};

// Used to draw a line between points
var linestring = {
    'type': 'Feature',
    'geometry': {
        'type': 'LineString',
        'coordinates': []
    }
};

/**
 * @see https://docs.mapbox.com/mapbox-gl-js/example/measure/
 */
function setPoint() {
    try {
        let center = map.getCenter();
        var point = {
            'type': 'Feature',
            'geometry': {
                'type': 'Point',
                'coordinates': [center.lng, center.lat]
            },
            'properties': {
                'id': String(new Date().getTime())
            }
        };

        if (geojson.features.length > 1) geojson.features.pop();
        distanceContainer.innerHTML = '';

        geojson.features.push(point);


        if (geojson.features.length > 1) {
            linestring.geometry.coordinates = geojson.features.map(function (
                point
            ) {
                return point.geometry.coordinates;
            });

            geojson.features.push(linestring);

            // Populate the distanceContainer with total distance
            var value = document.createElement('pre');
            value.textContent =
                'Total distance: ' +
                turf.length(linestring).toLocaleString() +
                'km';
            distanceContainer.appendChild(value);
        }

        console.log(geojson);


        map.getSource('geojson').setData(geojson);
    } catch (e) {
        geojson = {
            'type': 'FeatureCollection',
            'features': []
        };

        linestring = {
            'type': 'Feature',
            'geometry': {
                'type': 'LineString',
                'coordinates': []
            }
        };
    }
}

function activateDistanceMeasurements() {
    map.addSource('geojson', {
        'type': 'geojson',
        'data': geojson
    });

    // Add styles to the map
    map.addLayer({
        id: 'measure-points',
        type: 'circle',
        source: 'geojson',
        paint: {
            'circle-radius': 5,
            'circle-color': '#000'
        },
        filter: ['in', '$type', 'Point']
    });
    map.addLayer({
        id: 'measure-lines',
        type: 'line',
        source: 'geojson',
        layout: {
            'line-cap': 'round',
            'line-join': 'round'
        },
        paint: {
            'line-color': '#000',
            'line-width': 2.5
        },
        filter: ['in', '$type', 'LineString']
    });

}

function deactivateDistanceMeasurements() {
    map.removeLayer("measure-points");
    map.removeLayer("measure-lines");
    map.removeSource("geojson");
    geojson = {
        'type': 'FeatureCollection',
        'features': []
    }
    linestring = {
        'type': 'Feature',
        'geometry': {
            'type': 'LineString',
            'coordinates': []
        }
    }
    distanceContainer.innerHTML = "";
}

/*
MAP NAVIGATION
*/

function rotate(bearing) {
    let curBearing = map.getBearing();
    map.rotateTo(curBearing + bearing);
}

function pitch(pitch) {
    map.easeTo({
        pitch: pitch
    });
}

function resetView() {
    map.resetNorthPitch({
        duration: ANIMATION_DURATION
    });
}

function up() {
    let bounds = map.getBounds();
    let newCenter = new mapboxgl.LngLat(bounds.getCenter().lng, bounds.getNorth());

    map.panTo(newCenter, {
        duration: ANIMATION_DURATION
    });
}

function down() {
    let bounds = map.getBounds();
    let newCenter = new mapboxgl.LngLat(bounds.getCenter().lng, bounds.getSouth());

    map.panTo(newCenter, {
        duration: ANIMATION_DURATION
    });
}

function left() {
    let bounds = map.getBounds();
    let newCenter = new mapboxgl.LngLat(bounds.getWest(), bounds.getCenter().lat);

    map.panTo(newCenter, {
        duration: ANIMATION_DURATION
    });
}

function right() {
    let bounds = map.getBounds();
    let newCenter = new mapboxgl.LngLat(bounds.getEast(), bounds.getCenter().lat);

    map.panTo(newCenter, {
        duration: ANIMATION_DURATION
    });
}

function upRight() {
    let bounds = map.getBounds();
    map.panTo(bounds.getNorthEast(), {
        duration: ANIMATION_DURATION
    })
}

function upLeft() {
    let bounds = map.getBounds();
    map.panTo(bounds.getNorthWest(), {
        duration: ANIMATION_DURATION
    })
}

function downRight() {
    let bounds = map.getBounds();
    map.panTo(bounds.getSouthEast(), {
        duration: ANIMATION_DURATION
    })
}

function downLeft() {
    let bounds = map.getBounds();
    map.panTo(bounds.getSouthWest(), {
        duration: ANIMATION_DURATION
    })
}

function zoomIn() {
    map.zoomIn({
        duration: ANIMATION_DURATION
    })
}

function zoomOut() {
    map.zoomOut({
        duration: ANIMATION_DURATION
    })
}