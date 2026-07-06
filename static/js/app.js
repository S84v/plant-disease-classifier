const imageInput = document.getElementById("image-input");
const imagePreview = document.getElementById("image-preview");
const previewPlaceholder = document.getElementById("preview-placeholder");
const predictButton = document.getElementById("predict-button");

// state
let selectedFile = null;

// handle image selection
imageInput.addEventListener("change", (event) => {
    const file = event.target.files[0];

    if (!file) return;

    selectedFile = file;

    // validate type (basic safety)
    if (!file.type.startsWith("image/")) {
        alert("Please select a valid image file.");
        resetUI();
        return;
    }

    const reader = new FileReader();

    reader.onload = (e) => {
        imagePreview.src = e.target.result;

        // show image, hide placeholder
        imagePreview.classList.remove("d-none");
        previewPlaceholder.classList.add("d-none");

        // enable button
        predictButton.disabled = false;
    };

    reader.readAsDataURL(file);
});

// reset helper (optional but clean)
function resetUI() {
    selectedFile = null;
    imageInput.value = "";
    imagePreview.src = "";

    imagePreview.classList.add("d-none");
    previewPlaceholder.classList.remove("d-none");

    predictButton.disabled = true;
}