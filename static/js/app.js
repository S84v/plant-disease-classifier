const imageInput = document.getElementById("image-input");
const imagePreview = document.getElementById("image-preview");
const previewPlaceholder = document.getElementById("preview-placeholder");
const predictButton = document.getElementById("predict-button");

const loadingSection = document.getElementById("loading-section");
const resultSection = document.getElementById("result-section");
const diseaseInfoSection = document.getElementById("disease-info-section");
const errorSection = document.getElementById("error-section");

const predictedClassEl = document.getElementById("predicted-class");
const confidenceScoreEl = document.getElementById("confidence-score");
const topPredictionsEl = document.getElementById("top-predictions");

const descriptionEl = document.getElementById("description");
const symptomsEl = document.getElementById("symptoms");
const treatmentEl = document.getElementById("treatment");
const preventionEl = document.getElementById("prevention");
const errorMessageEl = document.getElementById("error-message");

// state
let selectedFile = null;

// ================= IMAGE SELECTION =================
imageInput.addEventListener("change", (event) => {
    const file = event.target.files[0];

    if (!file) return;

    if (!file.type.startsWith("image/")) {
        alert("Please upload a valid image file.");
        resetUI();
        return;
    }

    selectedFile = file;

    const reader = new FileReader();

    reader.onload = (e) => {
        imagePreview.src = e.target.result;

        imagePreview.classList.remove("d-none");
        previewPlaceholder.classList.add("d-none");

        predictButton.disabled = false;
    };

    reader.readAsDataURL(file);
});

// ================= PREDICT =================
predictButton.addEventListener("click", async () => {
    if (!selectedFile) return;

    clearUI();
    showLoading(true);

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
        const response = await fetch("/predict", {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || "Prediction failed");
        }

        const data = await response.json();

        renderResults(data);

    } catch (error) {
        showError(error.message);
    } finally {
        showLoading(false);
    }
});

// ================= RENDER RESULTS =================
function renderResults(data) {

    resultSection.classList.remove("d-none");
    diseaseInfoSection.classList.remove("d-none");

    // ================= MAIN PREDICTION =================
    predictedClassEl.textContent = data.predicted_class;

    confidenceScoreEl.textContent =
        (data.confidence * 100).toFixed(2) + "%";

    // ================= TOP-K =================
    topPredictionsEl.innerHTML = "";

    data.top_k_predictions.forEach((item, index) => {

        const confidence = item.confidence * 100;

        let barColor = "#dc3545"; // red default

        if (confidence >= 70) {
            barColor = "#198754"; // green
        } else if (confidence >= 40) {
            barColor = "#ffc107"; // yellow
        }

        const li = document.createElement("li");

        li.className = "list-group-item";

        const isTop = index === 0;

        li.innerHTML = `
        <div class="${isTop ? 'top-pred' : ''}">
            ${item.class_name}
        </div>

        <div class="text-muted small">
            ${confidence.toFixed(2)}%
        </div>

        <div class="conf-bar">
            <div class="conf-bar-fill"
                 style="width: ${confidence}%; background: ${barColor};">
            </div>
        </div>
    `;

        topPredictionsEl.appendChild(li);
    });

    // ================= DISEASE INFO =================
    const info = data.disease_info;

    descriptionEl.textContent = info.description || "N/A";

    symptomsEl.innerHTML = (info.symptoms || [])
        .map(s => `<li>${s}</li>`).join("");

    treatmentEl.innerHTML = (info.treatment || [])
        .map(t => `<li>${t}</li>`).join("");

    preventionEl.innerHTML = (info.prevention || [])
        .map(p => `<li>${p}</li>`).join("");
}

// ================= UI STATES =================
function showLoading(state) {
    loadingSection.classList.toggle("d-none", !state);
    predictButton.disabled = state;
}

function showError(message) {
    errorSection.classList.remove("d-none");
    errorMessageEl.textContent = message;
}

function clearUI() {
    resultSection.classList.add("d-none");
    diseaseInfoSection.classList.add("d-none");
    errorSection.classList.add("d-none");
}

function resetUI() {
    selectedFile = null;
    imageInput.value = "";

    imagePreview.src = "";
    imagePreview.classList.add("d-none");
    previewPlaceholder.classList.remove("d-none");

    predictButton.disabled = true;

    clearUI();
}