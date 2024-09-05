function fetchGeminiOutput() {
    fetch("/calculate")
      .then((response) => response.text())
      .then((data) => {
        document.getElementById("gemini-output").innerText = data;
      });
  }