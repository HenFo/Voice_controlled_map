var SpeechRecognition = SpeechRecognition || webkitSpeechRecognition;
var SpeechRecognitionEvent = SpeechRecognitionEvent || webkitSpeechRecognitionEvent;

var speechElement = new SpeechRecognition();
speechElement.lang = 'de-DE';
speechElement.interimResults = true;
speechElement.continuous = false;
var final_transcript = "";

let statusDOM = document.getElementById("status");
statusDOM.innerHTML = "not listening";
let curSpeechDOM = document.getElementById("curSpeech");
let speechResultDOM = document.getElementById("speechResult");
let textDOM = document.getElementById("command");
let statusColorDOM = document.getElementById("statusColor");

function startListening() {
    speechElement.start();
}

function stopListening() {
    speechElement.abort();
}

speechElement.onstart = function () {
    statusDOM.innerHTML = "listening";
    statusColorDOM.style = "background-color: green;";
}

speechElement.onend = function () {
    statusDOM.innerHTML = "not listening";
    textDOM.value = final_transcript;
    statusColorDOM.style = "background-color: red;";
}


speechElement.onresult = function (event) {
    var interim_transcript = "";
    for (let i = event.resultIndex; i < event.results.length; ++i) {
        let transcript = event.results[i][0].transcript;
        if (event.results[i].isFinal) {
            final_transcript = transcript;

        } else {
            interim_transcript += transcript;
        }
    }
    curSpeechDOM.innerHTML = interim_transcript;
    speechResultDOM.innerHTML = final_transcript;

}

speechElement.onspeechstart = function () {
    statusColorDOM.style = "background-color: orange;";
}
speechElement.onspeechend = function () {
    statusColorDOM.style = "background-color: green;";
}