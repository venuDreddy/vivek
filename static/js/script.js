function startPreprocess() {
  const button = document.getElementById("preprocess-button");
  button.classList.add("processing");

  fetch("/preprocess", {
    method: "POST",
  })
    .then((response) => response.json())
    .then((data) => {
      alert(data.message);
      button.classList.remove("processing");
    });
}

function calculateAccuracies() {
  const button = document.getElementById("accuracy-button");
  button.classList.add("processing");

  fetch("/calculate_accuracies", {
    method: "POST",
  })
    .then((response) => response.json())
    .then((data) => {
      let output = "";
      for (const [model, accuracy] of Object.entries(data)) {
        output += `${model}: ${accuracy}%<br>`;
      }
      document.getElementById("prediction-output").innerHTML = output;
      button.classList.remove("processing");
    });
}

function clearForm() {
  document.getElementById("input-form").reset();
  document.getElementById("prediction-output").innerHTML = "";

  // Ensure the diet image is removed properly
  const dietImage = document.getElementById("diet-image");
  dietImage.style.display = "none";

  // Force a refresh by assigning a blank image with a timestamp
  dietImage.src =
    "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///ywAAAAAAQABAAACAUwAOw==";
}

document
  .getElementById("input-form")
  .addEventListener("submit", function (event) {
    event.preventDefault();

    const button = document.getElementById("predict-button");
    button.classList.add("processing");

    const formData = new FormData(event.target);
    const data = Object.fromEntries(formData.entries());

    fetch("/predict", {
      method: "POST",
      body: new URLSearchParams(data),
    })
      .then((response) => response.json())
      .then((data) => {
        document.getElementById(
          "prediction-output"
        ).innerHTML = `Prediction: ${data.outcome}`;

        const dietImage = document.getElementById("diet-image");
        if (data.diet_image) {
          dietImage.src = data.diet_image + "?" + new Date().getTime(); // Force reload
          dietImage.style.display = "block";
        } else {
          dietImage.style.display = "none";
          dietImage.src =
            "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///ywAAAAAAQABAAACAUwAOw==";
        }

        button.classList.remove("processing");
      });
  });
