const audioContext = new (window.AudioContext || window.webkitAudioContext)();
let oscillators = [];
let playing = false;

document.getElementById('add-oscillator').addEventListener('click', addOscillator);
document.getElementById('play').addEventListener('click', playSound);
document.getElementById('stop').addEventListener('click', stopSound);

function addOscillator() {
    const oscIndex = oscillators.length;
    const oscDiv = document.createElement('div');
    oscDiv.classList.add('oscillator');

    const freqLabel = document.createElement('label');
    freqLabel.textContent = `Frequency: `;
    const freqInput = document.createElement('input');
    freqInput.type = 'range';
    freqInput.min = 50;
    freqInput.max = 1000;
    freqInput.value = 440;

    const gainLabel = document.createElement('label');
    gainLabel.textContent = `Gain: `;
    const gainInput = document.createElement('input');
    gainInput.type = 'range';
    gainInput.min = 0;
    gainInput.max = 1;
    gainInput.step = 0.01;
    gainInput.value = 0.5;

    oscDiv.appendChild(freqLabel);
    oscDiv.appendChild(freqInput);
    oscDiv.appendChild(gainLabel);
    oscDiv.appendChild(gainInput);

    document.getElementById('oscillators').appendChild(oscDiv);

    const oscillator = audioContext.createOscillator();
    const gainNode = audioContext.createGain();

    oscillator.frequency.value = freqInput.value;
    gainNode.gain.value = gainInput.value;

    freqInput.addEventListener('input', (e) => {
        oscillator.frequency.value = e.target.value;
    });

    gainInput.addEventListener('input', (e) => {
        gainNode.gain.value = e.target.value;
    });

    oscillator.connect(gainNode).connect(audioContext.destination);
    oscillators.push({ oscillator, gainNode });
}

function playSound() {
    if (playing) {
        return;
    }

    if(oscillators.length === 0) return;

    console.log('For each oscillator...');
    oscillators.forEach(({ oscillator }) => {
        oscillator.start();
    });

    playing = true;
}

function stopSound() {
    if (!playing) return;
    oscillators.forEach(({ oscillator }) => {
        oscillator.stop();
    });
    oscillators = [];
    playing = false;
}

function startStopDelegator() {
    if(!playing) {
        oscillators.forEach(({ oscillator }) => {
            oscillator.start();
        });
        playing = true;
    }
    else {
        oscillators.forEach(({ oscillator }) => {
            oscillator.stop();
        });
        oscillators = [];
        playing = false;
    }
}
