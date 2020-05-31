/**
 * sets js to sleep
 * function that uses sleep must be declared async
 * @param {Number} milliseconds
 * @see https://www.sitepoint.com/delay-sleep-pause-wait/
 */
function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

/* 
0/1 = basic direction; 2/3 = combined direction
*/
var state = 24;
var used = -1;
let repeat = false;
async function animateMap() {
    console.log(state);
    let button1 = document.getElementById("animateMap");
    let button2 = document.getElementById("repeatMap");
    button1.disabled = "disabled";
    button2.disabled = "disabled";
    switch (state) {
        case 0:
        case 1: {
            let selec = -1;
            if (!repeat) {
                do {
                    selec = Math.round(Math.random() * 3);
                } while (selec == used)
            } else {
                selec = used;
            }
            used = selec;
            switch (selec) {
                case 0:
                    up();
                    await sleep(4000);
                    resetPosition();
                    break;
                case 1:
                    right();
                    await sleep(4000);
                    resetPosition();
                    break;
                case 2:
                    down();
                    await sleep(4000);
                    resetPosition();
                    break;
                case 3:
                    left();
                    await sleep(4000);
                    resetPosition();
                    break;
                default:
                    break;
            }
            state++;
            break;
        }

        case 2:
        case 3: {
            let selec = -1;
            if (!repeat) {
                do {
                    selec = Math.round(Math.random() * 3 + 4);
                } while (selec == used)
            } else {
                selec = used;
            }
            used = selec;
            switch (selec) {
                case 4:
                    upRight();
                    await sleep(4000);
                    resetPosition();
                    break;
                case 5:
                    downRight();
                    await sleep(4000);
                    resetPosition();
                    break;
                case 6:
                    downLeft();
                    await sleep(4000);
                    resetPosition();
                    break;
                case 7:
                    downLeft();
                    await sleep(4000);
                    resetPosition();
                    break;
                default:
                    break;
            }
            state++;
            break;
        }

        case 4: {
            zoomIn();
            await sleep(4000);
            resetPosition();
            state++;
            break;
        }

        case 5: {
            zoomOut();
            await sleep(4000);
            resetPosition();
            state++;
            break;
        }

        case 6: {
            rotate(45);
            await sleep(4000);
            resetPosition();
            state++;
            break;
        }
        case 7: {
            rotate(90);
            state++;
            break;
        }
        case 8: {
            pitch(45);
            state++;
            break;
        }
        case 9: {
            resetView();
            state++;
            break;
        }

        // Ansicht zurück auf Startposition
        case 10: {
            // resetPosition();
            // await sleep(4000);
            addMarker();
            await sleep(2000)
            map.panTo([-74.3, 40.1])
            await sleep(2000);
            addMarker();
            await sleep(2000);
            map.panTo([-74.7, 40.15])
            await sleep(2000);
            addMarker();
            state++;
            break;
        }
        case 11: {
            clearMarker();
            state++;
            break;
        }
        case 12: {
            showMenu();
            state++;
            break;
        }
        case 13: {
            hideMenu();
            state++;
            break;
        }
        case 14: {
            showHelp();
            state++;
            break;
        }
        case 15: {
            hideHelp();
            state++;
            break;
        }
        case 16: {
            showSearch();
            state++;
            break;
        }
        case 17: {
            hideSearch();
            state++;
            break;
        }
        case 18: {
            fakeUserLocation();
            state++;
            break;
        }
        case 19: {
            sateliteBasemap();
            state++;
            break;
        }
        case 20: {
            vectorBasemap();
            state++;
            break;
        }
        //auf Startansicht reseten
        case 21: {
            resetPosition();
            await sleep(3000);
            activateDistanceMeasurements();
            setPoint();
            await sleep(2000);
            map.panTo([-74.7, 40.15]);
            await sleep(2000);
            setPoint();
            await sleep(2000);
            map.panTo([-74.3, 40.1]);
            await sleep(2000);
            setPoint();
            await sleep(5000);
            deactivateDistanceMeasurements();
            state++;
            break;
        }

        case 22: {
            initPrevNext();
            await sleep(5000);
            next();
            await sleep(5000);
            next();
            await sleep(5000);
            state++;
            break;
        }

        case 23: {
            prev();
            await sleep(5000);
            prev();
            await sleep(5000);
            state++;
            break;
        }

        case 24: {
            let obj = document.getElementById("accept-reject");
            obj.style.visibility = "visible";
            await sleep(3000);
            await aniAcc();
            await sleep(1000);
            obj.style.visibility = "";
            let popup = document.getElementById("positiv");
            popup.style.visibility = "";
            popup.style.top = "";

            state++;
            break;
        }
        case 25: {
            let obj = document.getElementById("accept-reject");
            obj.style.visibility = "visible";

            await sleep(3000);
            await aniRej();
            // await sleep(1000);
            obj.style.visibility = "";
            let popup = document.getElementById("negativ");
            popup.style.visibility = "";
            popup.style.top = "";

            state++;
            break;
        }

        //////////////////
        default:
            state = 0;
            resetPosition();
            break;

    }
    button1.disabled = "";
    button2.disabled = "";

}

async function repeatMap() {
    switch (--state) {
        case 0:
        case 1:
        case 2:
        case 3: {
            repeat = true;
            break;
        }

        case 7: {
            resetPosition();
            await sleep(3000);
            break;
        }
        case 8: {
            map.jumpTo({
                pitch: 0,
                bearing: 90
            });
            await sleep(3000);
            break;
        }
        case 9: {
            map.jumpTo({
                pitch: 45,
                bearing: 90
            });
            await sleep(3000);
            break;
        }
        case 10: {
            resetPosition();
            clearMarker();
            await sleep(3000);
            break;
        }
        case 11: {
            let pos = [
                [-74.5, 40],
                [-74.3, 40.1],
                [-74.7, 40.15]
            ];
            for (let i = 0; i < pos.length; i++) {
                let marker = new mapboxgl.Marker()
                    .setLngLat(pos[i])
                    .addTo(map);
                markers.push(marker);
            }
            await sleep(1000);
            break;
        }
        case 19: {
            vectorBasemap();
            await sleep(5000);
            break;
        }
        case 20: {
            sateliteBasemap();
            await sleep(5000);
            break;
        }
        case 22: {
            let objs = document.querySelectorAll("#resultList > li");
            objs.forEach(li => {
                li.style.fontWeight = "";
                li.style.color = "";
            });
            stopNextPrev();
            break;
        }
        case 23: {
            stopNextPrev();
            initPrevNext();
            position = 2;
            let obj = document.querySelector(`#resultList > li:nth-child(${1})`);
            obj.style.fontWeight = "";
            obj.style.color = "";

            obj = document.querySelector(`#resultList > li:nth-child(${position+1})`);
            obj.style.fontWeight = "bolder";
            obj.style.color = "#525252";

            map.jumpTo({
                center: marker[position].getLngLat(),
                zoom: 12,
            });
            await sleep(3000);
            break;
        }


        default:
            break;
    }
    animateMap();
    repeat = false;
}