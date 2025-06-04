window.onload = async () => {
    let recorder, audioChunks = [];
    let audioBlob, audioUrl;
    const startBtn = document.getElementById("startRecord");
    const stopBtn = document.getElementById("stopRecord");
    const recordWaveDiv = document.getElementById("recordWave");
    const ttsWaveDiv = document.getElementById("ttsWave");
    let hasRecord = false;
    const response = await fetch("/speakers", {
      method: "GET",
    });

    const data = await response.json();
    var all='';
    data.data.forEach(element => {
        all+=`<option value="${element}">${element}</option>`
    });
    
	
    document.getElementById("voices").innerHTML = all

    const recordWave = WaveSurfer.create({
      container: recordWaveDiv,
      waveColor: "#93c5fd",
      progressColor: "#3b82f6",
      height: 128,
    });

  const ttsWave = WaveSurfer.create({
    container: ttsWaveDiv,
    waveColor: "#0083e0",
    progressColor: "#57b9ff",
    height: 128,
  });

  startBtn.onclick = async () => {
    startBtn.style.display = 'none';
    stopBtn.style.display = 'block';
    document.getElementById("mic-icon").classList.add("mic-animate");

    audioChunks = [];
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    recorder = new MediaRecorder(stream);

    recorder.ondataavailable = e => audioChunks.push(e.data);

    recorder.onstop = () => {
      hasRecord = true;
      //document.getElementById("micro").style.display = "none";
      //document.getElementById("recordWave").style.display = "block";
      document.getElementById("mic-icon").classList.remove("mic-animate");
      document.getElementById("mic-icon").style.opacity=0.3

      audioBlob = new Blob(audioChunks, { type: "audio/wav" });
      audioUrl = URL.createObjectURL(audioBlob);
      recordWave.load(audioUrl);
      document.getElementById("recordedAudio").src = audioUrl;
      document.getElementById("recordedAudio").play();

      stopBtn.style.display = 'none';
      document.getElementById('playRecord').style.display = 'block';
      document.getElementById('removeRecord').style.display = 'block';
      document.getElementById('btnTranscribe').style.display = 'block';
    };

    recorder.start();
  };

  stopBtn.onclick = () => {
    recorder.stop();
  };

  document.getElementById('playRecord').onclick = () => {
    recordWave.playPause();
  };

  document.getElementById('removeRecord').onclick = () => {
    recordWave.stop();
    hasRecord = false;
    document.getElementById('playRecord').style.display = 'none';
    document.getElementById('removeRecord').style.display = 'none';
    document.getElementById("micro").style.display = "block";
    
    document.getElementById("startRecord").style.display = "block";
    document.getElementById("btnTranscribe").style.display = "none";
  };

  document.getElementById("playTTS").onclick = () => {
    ttsWave.playPause()
  };
  document.getElementById('btnTranscribe').onclick = async () => {
    if (!audioBlob) {
      alert("Primero graba un audio");
      return;
    }

    const formData = new FormData();
    formData.append("audio", audioBlob, "grabacion.wav");
    formData.append("speaker",document.getElementById("voices").value);
    document.getElementById("cargando").classList.remove("hidden");
    try {
      const response = await fetch("/transcribe", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
       console.log("Response",data)
      document.getElementById("txtTranscripcion").innerText = data.text;
      const entityList = data.entities.map(ent => `${ent[0]} → ${ent[1]}`).join(", ");
      
      document.getElementById("txtEntitys").innerText = entityList;
      document.getElementById("txtTranslation").innerText = data.translation;
      ttsWave.load(data.url);
      document.getElementById("playTTS").style.display = "block";
      document.getElementById("cargando").classList.add("hidden");

    } catch (err) {
      console.error("Error al enviar audio:", err);
      document.getElementById("cargando").classList.add("hidden");
      toastr.error('Spekaer no disponible o error desconocido', 'Error!')
    }
  
  }
};
