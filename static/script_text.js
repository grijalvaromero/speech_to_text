window.onload = async () => {
    //GET VOICES
    const ttsWaveDiv = document.getElementById("ttsWave");
    const response = await fetch("/speakers", {method: "GET"});
    const data = await response.json();
    const ttsWave = WaveSurfer.create({
            container: ttsWaveDiv,
            waveColor: "#0083e0",
            progressColor: "#57b9ff",
            height: 128,
        });
    var all='';
    data.data.forEach(element => {
        all+=`<option value="${element}">${element}</option>`
    });
    document.getElementById("voices").innerHTML = all
    console.log(data)

    //GET TEXT FROM URL
    document.querySelector("#btnExtract").onclick=async()=>{
        document.getElementById("cargando").classList.remove("hidden");
        url =document.querySelector("#txtUrl").value
        if(url==''){
            toastr.error('Ingresa un enlace valido ', 'Error!')
            return 
        }
        const formData = new FormData();
        formData.append("url", url);
        const response = await fetch("/wiki", {method: "POST", body: formData});
        const data = await response.json();
        document.querySelector("#divText").innerHTML = data.data
        document.getElementById('btnTranscribe').style.display = 'block';
        document.getElementById("cargando").classList.add("hidden");
    }
    document.getElementById('btnTranscribe').onclick = async () => {
        text=document.querySelector("#divText").innerText
    
        console.log(text)
        if (text=='') {
            alert("Primero graba un audio");
            return;
        }
        
        const formData = new FormData();
        formData.append("text", text);
        formData.append("speaker",document.getElementById("voices").value);
        document.getElementById("cargando").classList.remove("hidden");
 
        try {
        const response = await fetch("/onlytext", {
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

    document.getElementById("playTTS").onclick = () => {
        ttsWave.playPause()
    };
}