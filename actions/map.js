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


async function doStuff() {
    activateDistanceMeasurements();
}

function doStuff2() {
    deactivateDistanceMeasurements();
}

function doStuff3() {
    setPoint();
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

function selectResult(number) {
    clearSelection();
    $(`#resultList li:nth-child(${number})`).css("background-color", "gray");
}

function clearSelection() {
    $("#resultList").children().css("background-color", "unset");
}

/*
MAP MANIPULATION
*/
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

        map.getSource('geojson').setData(geojson);
    } catch(e) {
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
    map.rotateTo(bearing);
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