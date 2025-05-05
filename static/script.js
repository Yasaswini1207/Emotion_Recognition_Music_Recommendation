setInterval(() => {
    fetch('/get_emotion')
    .then(response => response.json())
    .then(data => {
        document.getElementById('emotion_display').innerText = "Detected Emotion: " + data.emotion;
    });
}, 2000);
