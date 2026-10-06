const fileInput = document.querySelector("#fileInput");
const chooseButton = document.querySelector("#chooseButton");
const uploadPanel = document.querySelector("#uploadPanel");
const editor = document.querySelector("#editor");
const canvas = document.querySelector("#canvas");
const ctx = canvas.getContext("2d", { willReadFrequently: true });
const imageStatus = document.querySelector("#imageStatus");
const originalImage = new Image();
let sourceUrl = "";
let sourceData;
let backgroundPreview;
let backgroundBase;
let segmentationModel;
let segmentationProcessor;

const controls = ["lighting", "color", "warmth", "clarity"];
const values = Object.fromEntries(controls.map((name) => [name, document.querySelector(`#${name}`)]));
const backgroundStyle = document.querySelector("#backgroundStyle");
const backgroundDescription = document.querySelector("#backgroundDescription");
const backgroundPresets = {
  original: { description: "AI keeps the original background" },
  linen: { description: "AI places the products on soft linen" },
  wood: { description: "AI places the products on warm wood" },
  gray: { description: "AI places the products on a clean studio background" },
};

chooseButton.addEventListener("click", () => fileInput.click());
document.querySelector("#newImageButton").addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", (event) => loadFile(event.target.files[0]));

["dragenter", "dragover"].forEach((eventName) => uploadPanel.addEventListener(eventName, (event) => {
  event.preventDefault();
  uploadPanel.classList.add("dragging");
}));
["dragleave", "drop"].forEach((eventName) => uploadPanel.addEventListener(eventName, (event) => {
  event.preventDefault();
  uploadPanel.classList.remove("dragging");
}));
uploadPanel.addEventListener("drop", (event) => loadFile(event.dataTransfer.files[0]));

controls.forEach((name) => values[name].addEventListener("input", () => {
  document.querySelector(`#${name}Value`).textContent = values[name].value;
  render();
}));
backgroundStyle.addEventListener("change", () => {
  backgroundDescription.textContent = backgroundPresets[backgroundStyle.value].description;
  render();
});
document.querySelector("#applyBackgroundButton").addEventListener("click", applyBackground);
document.querySelector("#acceptBackgroundButton").addEventListener("click", () => {
  backgroundPreview = ctx.getImageData(0, 0, canvas.width, canvas.height);
  document.querySelector("#backgroundReview").classList.add("hidden");
  imageStatus.textContent = "Background replaced";
});
document.querySelector("#revertBackgroundButton").addEventListener("click", () => {
  if (backgroundBase) ctx.putImageData(backgroundBase, 0, 0);
  backgroundPreview = null;
  document.querySelector("#backgroundReview").classList.add("hidden");
  render();
});

document.querySelector("#presetButton").addEventListener("click", () => {
  const preset = { lighting: 14, color: 5, warmth: 1, clarity: 8 };
  controls.forEach((name) => {
    values[name].value = preset[name];
    document.querySelector(`#${name}Value`).textContent = preset[name];
  });
  render();
});
document.querySelector("#resetButton").addEventListener("click", () => {
  const defaults = { lighting: 0, color: 0, warmth: 0, clarity: 0 };
  controls.forEach((name) => {
    values[name].value = defaults[name];
    document.querySelector(`#${name}Value`).textContent = defaults[name];
  });
  backgroundStyle.value = "original";
  backgroundDescription.textContent = backgroundPresets.original.description;
  render();
});

const compareButton = document.querySelector("#compareButton");
compareButton.addEventListener("pointerdown", () => {
  if (!sourceData) return;
  ctx.putImageData(sourceData, 0, 0);
  imageStatus.textContent = "Original";
});
["pointerup", "pointerleave"].forEach((eventName) => compareButton.addEventListener(eventName, () => {
  if (sourceData) { render(); imageStatus.textContent = "Enhanced"; }
}));

document.querySelector("#downloadButton").addEventListener("click", () => {
  const link = document.createElement("a");
  link.download = "retouch-product-photo.png";
  link.href = canvas.toDataURL("image/png");
  link.click();
});

function loadFile(file) {
  if (!file || !file.type.startsWith("image/")) return;
  if (sourceUrl) URL.revokeObjectURL(sourceUrl);
  sourceUrl = URL.createObjectURL(file);
  originalImage.onload = () => {
    const scale = Math.min(1, 1800 / originalImage.width, 1800 / originalImage.height);
    canvas.width = Math.round(originalImage.width * scale);
    canvas.height = Math.round(originalImage.height * scale);
    ctx.drawImage(originalImage, 0, 0, canvas.width, canvas.height);
    sourceData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    uploadPanel.classList.add("hidden");
    editor.classList.remove("hidden");
    render();
  };
  originalImage.src = sourceUrl;
}

function render() {
  if (!sourceData) return;
  const pixels = new Uint8ClampedArray(sourceData.data);
  const light = Number(values.lighting.value);
  const color = Number(values.color.value);
  const warmth = Number(values.warmth.value);
  const clarity = Number(values.clarity.value);
  const contrast = 1 + clarity / 160;
  for (let i = 0; i < pixels.length; i += 4) {
    const luminance = (pixels[i] * .299 + pixels[i + 1] * .587 + pixels[i + 2] * .114) / 255;
    const lift = light * (0.72 + (1 - luminance) * .55);
    pixels[i] = clamp((pixels[i] + lift + warmth) * contrast - 128 * (contrast - 1) + color * .3);
    pixels[i + 1] = clamp((pixels[i + 1] + lift) * contrast - 128 * (contrast - 1) + color * .15);
    pixels[i + 2] = clamp((pixels[i + 2] + lift - warmth) * contrast - 128 * (contrast - 1) - color * .2);
  }
  ctx.putImageData(new ImageData(pixels, canvas.width, canvas.height), 0, 0);
  imageStatus.textContent = "Enhanced";
}

async function applyBackground() {
  if (!sourceData || backgroundStyle.value === "original") return;
  const button = document.querySelector("#applyBackgroundButton");
  button.disabled = true;
  button.textContent = "Loading background remover…";
  try {
    backgroundBase = ctx.getImageData(0, 0, canvas.width, canvas.height);
    button.textContent = "Loading BiRefNet model (first use is large)…";
    const transformers = await import("https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.2/+esm");
    if (!segmentationModel) {
      segmentationModel = await transformers.AutoModel.from_pretrained("onnx-community/BiRefNet-ONNX", { dtype: "fp32" });
      segmentationProcessor = await transformers.AutoProcessor.from_pretrained("onnx-community/BiRefNet-ONNX");
    }
    button.textContent = "Removing background…";
    const image = await transformers.RawImage.fromURL(canvas.toDataURL("image/png"));
    const { pixel_values: pixelValues } = await segmentationProcessor(image);
    const { output_image: outputImage } = await segmentationModel({ input_image: pixelValues });
    const mask = await transformers.RawImage.fromTensor(outputImage[0].sigmoid().mul(255).to("uint8")).resize(canvas.width, canvas.height);
    const blob = await createTransparentForeground(mask);
    const foregroundUrl = URL.createObjectURL(blob);
    const foreground = new Image();
    foreground.onload = () => {
      drawBackgroundSurface(backgroundStyle.value);
      drawContactShadow(foreground);
      ctx.drawImage(foreground, 0, 0, canvas.width, canvas.height);
      URL.revokeObjectURL(foregroundUrl);
      backgroundPreview = ctx.getImageData(0, 0, canvas.width, canvas.height);
      document.querySelector("#backgroundReview").classList.remove("hidden");
      imageStatus.textContent = "Background preview";
      button.disabled = false;
      button.textContent = "Apply AI background";
    };
    foreground.src = foregroundUrl;
  } catch (error) {
    console.error("Background removal failed", error);
    button.disabled = false;
    button.textContent = "Try background again";
    imageStatus.textContent = "Background removal failed";
  }
}

async function createTransparentForeground(mask) {
  const pixels = new Uint8ClampedArray(sourceData.data);
  const maskData = mask.data;
  for (let i = 0; i < pixels.length; i += 4) {
    const maskIndex = Math.floor(i / 4);
    pixels[i + 3] = maskData[maskIndex * (mask.channels || 1)] ?? 0;
  }
  const foregroundCanvas = document.createElement("canvas");
  foregroundCanvas.width = canvas.width;
  foregroundCanvas.height = canvas.height;
  foregroundCanvas.getContext("2d").putImageData(new ImageData(pixels, canvas.width, canvas.height), 0, 0);
  return new Promise((resolve, reject) => foregroundCanvas.toBlob((blob) => blob ? resolve(blob) : reject(new Error("Could not create foreground")), "image/png"));
}

function canvasToBlob() {
  return new Promise((resolve, reject) => canvas.toBlob((blob) => blob ? resolve(blob) : reject(new Error("Could not read image")), "image/png"));
}

function drawBackgroundSurface(style) {
  const colors = { linen: ["#d9d0bf", "#bdb19c"], wood: ["#ad7548", "#70472e"], gray: ["#f7f7f5", "#d7d8d5"] };
  const [top, bottom] = colors[style];
  const gradient = ctx.createLinearGradient(0, 0, 0, canvas.height);
  gradient.addColorStop(0, top);
  gradient.addColorStop(1, bottom);
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  if (style === "wood") drawWoodGrain();
  if (style === "linen") drawLinenTexture();
}

function drawContactShadow(foreground) {
  const shadow = document.createElement("canvas");
  shadow.width = canvas.width;
  shadow.height = canvas.height;
  const shadowContext = shadow.getContext("2d");
  shadowContext.filter = "blur(14px)";
  shadowContext.globalAlpha = 0.22;
  shadowContext.drawImage(foreground, 0, 12, canvas.width, canvas.height);
  ctx.globalAlpha = 0.35;
  ctx.drawImage(shadow, 0, 0);
  ctx.globalAlpha = 1;
}

function drawWoodGrain() {
  ctx.strokeStyle = "rgba(74,45,26,.18)";
  ctx.lineWidth = 1;
  for (let y = 18; y < canvas.height; y += 34) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.bezierCurveTo(canvas.width * .25, y - 8, canvas.width * .7, y + 8, canvas.width, y - 2);
    ctx.stroke();
  }
}

function drawLinenTexture() {
  ctx.strokeStyle = "rgba(120,110,90,.1)";
  ctx.lineWidth = 1;
  for (let x = 0; x < canvas.width; x += 12) {
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, canvas.height); ctx.stroke();
  }
}

function clamp(value) {
  return Math.max(0, Math.min(255, value));
}
