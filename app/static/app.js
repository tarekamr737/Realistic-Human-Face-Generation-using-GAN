(() => {
  "use strict";

  const state = { count: 4, modelAvailable: false, comparisonReady: false, models: [], result: null, preview: null };
  const $ = (selector) => document.querySelector(selector);
  const $$ = (selector) => [...document.querySelectorAll(selector)];
  const seedInput = $("#seed");
  const generateButton = $("#generateButton");
  const generationGrid = $("#generationGrid");
  const emptyGeneration = $("#emptyGeneration");
  const toast = $("#toast");
  let toastTimer;

  function notify(message, isError = false) {
    toast.textContent = message;
    toast.classList.toggle("is-error", isError);
    toast.classList.add("is-visible");
    clearTimeout(toastTimer);
    toastTimer = window.setTimeout(() => toast.classList.remove("is-visible"), 3200);
  }

  function setStatus(available, message, readyCount = 0) {
    state.modelAvailable = available;
    const dot = $("#headerStatus .status-dot");
    dot.classList.toggle("is-ready", available);
    dot.classList.toggle("is-unavailable", !available);
    $("#headerStatus span:last-child").textContent = available ? `${readyCount}/2 models ready` : "Models unavailable";
    $("#modelPanel .status-dot").classList.toggle("is-ready", available);
    $("#modelPanel .status-dot").classList.toggle("is-unavailable", !available);
    $("#modelAvailability").textContent = available ? "Inference ready" : "Generators unavailable";
    $("#modelMessage").textContent = message;
    generateButton.disabled = !available;
    $("#railNote").textContent = available
      ? "This app can only generate. Training and fine-tuning stay outside this web app."
      : "Install both local model assets to enable the complete comparison.";
  }

  async function api(path, options = {}) {
    const response = await fetch(path, { headers: { "Content-Type": "application/json" }, ...options });
    const body = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(body.detail || "The request could not be completed.");
    return body;
  }

  function renderModelAvailability(models) {
    const list = $("#modelAvailabilityList");
    list.replaceChildren(...models.map((model) => {
      const row = document.createElement("div");
      row.className = "model-availability-row";
      const name = document.createElement("span");
      name.textContent = model.id === "faceforge-dcgan-64" ? "Ours · DCGAN-64" : "Reference · R3GAN-256";
      const status = document.createElement("b");
      status.textContent = model.available ? "Ready" : "Missing";
      status.className = model.available ? "is-ready" : "is-missing";
      row.append(name, status);
      return row;
    }));
  }

  async function loadModel() {
    try {
      const data = await api("/api/models");
      state.models = data.models || [];
      state.comparisonReady = Boolean(data.comparison_ready);
      const readyCount = state.models.filter((model) => model.available).length;
      setStatus(Boolean(data.available), state.comparisonReady
        ? "Both generators are loaded for a reproducible side-by-side comparison."
        : "One or more optional model assets are missing; available generators can still run.", readyCount);
      const model = state.models.find((entry) => entry.id === "faceforge-dcgan-64") || {};
      $("#checkpointName").textContent = model.checkpoint || "generator_best.pt";
      $("#checkpointInfo").textContent = model.checkpoint || "generator_best.pt";
      $("#architecture").textContent = model.architecture || "DCGAN";
      $("#resolution").textContent = `${model.image_size || 64} × ${model.image_size || 64}`;
      $("#latentDim").textContent = `${model.latent_dim || 128}-D`;
      $("#datasetName").textContent = model.dataset || "FFHQ";
      $("#deviceInfo").textContent = model.device || "Local CPU";
      $("#comparisonStatus").textContent = data.comparison_note || "The same numeric seed is reproducible, but each model has its own latent space.";
      $("#comparisonStatus").classList.toggle("is-ready", state.comparisonReady);
      renderModelAvailability(state.models);
    } catch (error) {
      setStatus(false, "The local service could not be reached. Start the app with uvicorn, then try again.");
      $("#comparisonStatus").textContent = "Comparison status is unavailable because the local service could not be reached.";
      $("#modelAvailabilityList").replaceChildren();
    }
  }

  function validateSeed() {
    const value = seedInput.value.trim();
    const valid = /^\d{1,10}$/.test(value) && Number(value) <= 2147483647;
    $("#seedError").textContent = valid ? "" : "Seed needs to be a whole number from 0 to 2,147,483,647.";
    return valid ? Number(value) : null;
  }

  function randomSeed() {
    const numbers = new Uint32Array(1);
    crypto.getRandomValues(numbers);
    seedInput.value = String(numbers[0] % 2147483647);
    validateSeed();
    seedInput.focus();
  }

  function renderLoading() {
    emptyGeneration.hidden = true;
    const activeModels = Math.max(1, state.models.filter((model) => model.available).length);
    generationGrid.className = "comparison-results is-loading";
    generationGrid.replaceChildren(...Array.from({ length: activeModels }, (_, modelIndex) => {
      const column = document.createElement("article");
      column.className = "comparison-model skeleton-column";
      const title = document.createElement("div");
      title.className = "comparison-model-heading";
      title.textContent = "Generating synthetic samples";
      const grid = document.createElement("div");
      grid.className = "comparison-grid";
      grid.append(...Array.from({ length: Math.min(state.count, 4) }, (_, index) => {
        const tile = document.createElement("div");
        tile.className = "face-card skeleton";
        tile.style.setProperty("--index", index + modelIndex * state.count);
        tile.setAttribute("aria-label", "Generating synthetic face");
        return tile;
      }));
      column.append(title, grid);
      return column;
    }));
  }

  function download(dataUrl, filename) {
    const link = document.createElement("a");
    link.href = dataUrl;
    link.download = filename;
    document.body.append(link);
    link.click();
    link.remove();
  }

  async function copyText(text, confirmation) {
    try {
      await navigator.clipboard.writeText(String(text));
      notify(confirmation);
    } catch {
      notify("Copy is unavailable in this browser. Select the seed field to copy it.", true);
    }
  }

  function preview(card) {
    state.preview = card;
    const dialog = $("#previewDialog");
    $("#previewImage").src = card.image;
    $("#previewImage").alt = card.label;
    $("#previewTitle").textContent = `${card.modelName || "Synthetic face"} · seed ${card.seed}`;
    dialog.showModal();
  }

  function makeFaceCard(card, index, modelName) {
      const figure = document.createElement("figure");
      figure.className = "face-card";
      figure.style.setProperty("--index", index);
      card.modelName = modelName;
      const image = document.createElement("img");
      image.src = card.image;
      image.alt = card.label;
      image.loading = index > 3 ? "lazy" : "eager";
      image.setAttribute("role", "button");
      image.addEventListener("click", () => preview(card));
      image.tabIndex = 0;
      image.addEventListener("keydown", (event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); preview(card); } });
      const meta = document.createElement("figcaption");
      meta.className = "face-meta";
      meta.innerHTML = `<span class="synthetic-label">Synthetic</span><span class="face-actions"><button type="button" aria-label="Preview face with seed ${card.seed}" title="Preview">⌕</button><button type="button" aria-label="Download face with seed ${card.seed}" title="Download">↓</button></span>`;
      const [previewButton, downloadButton] = meta.querySelectorAll("button");
      previewButton.addEventListener("click", () => preview(card));
      downloadButton.addEventListener("click", () => download(card.image, card.filename));
      figure.append(image, meta);
      return figure;
  }

  function makeModelResult(model, modelIndex) {
    const column = document.createElement("article");
    column.className = "comparison-model";
    const heading = document.createElement("header");
    heading.className = "comparison-model-heading";
    const title = document.createElement("div");
    const name = document.createElement("h3");
    name.textContent = model.display_name;
    const meta = document.createElement("p");
    meta.textContent = `${model.architecture} · native ${model.image_size} × ${model.image_size}`;
    title.append(name, meta);
    const downloadButton = document.createElement("button");
    downloadButton.className = "secondary-button comparison-download";
    downloadButton.type = "button";
    downloadButton.textContent = "Download grid";
    downloadButton.addEventListener("click", () => download(model.grid, model.grid_filename));
    heading.append(title, downloadButton);
    const note = document.createElement("p");
    note.className = "comparison-note";
    note.textContent = model.sampling_note;
    const grid = document.createElement("div");
    grid.className = "comparison-grid";
    grid.append(...model.images.map((card, index) => makeFaceCard(card, index + modelIndex * state.count, model.display_name)));
    column.append(heading, note, grid);
    return column;
  }

  function renderResult(result) {
    emptyGeneration.hidden = true;
    generationGrid.className = "comparison-results";
    generationGrid.replaceChildren(...result.models.map((model, index) => makeModelResult(model, index)));
    $("#copySeed").disabled = false;
    $("#downloadGrid").disabled = false;
    $("#runInfo").textContent = `${result.models.length} model${result.models.length === 1 ? "" : "s"} · ${result.count} faces each · shared seed ${result.seed}`;
  }

  async function generate() {
    const seed = validateSeed();
    if (seed === null) return;
    if (!state.modelAvailable) { notify("A compatible local generator checkpoint is required before generation.", true); return; }
    generateButton.disabled = true;
    generateButton.innerHTML = "<span aria-hidden=\"true\">⋯</span> Generating comparison";
    renderLoading();
    try {
      const result = await api("/api/generate", { method: "POST", body: JSON.stringify({ count: state.count, seed, truncation: Number($("#truncation").value) }) });
      state.result = result;
      seedInput.value = String(result.seed);
      renderResult(result);
      notify(result.comparison_ready
        ? "Comparison generated. Both collections use the same numeric seed."
        : "Available model generated. Install the missing model to complete the comparison.");
    } catch (error) {
      generationGrid.replaceChildren();
      emptyGeneration.hidden = false;
      notify(error.message || "Generation failed. Confirm that the exported checkpoint is compatible, then try again.", true);
    } finally {
      generateButton.disabled = !state.modelAvailable;
      generateButton.innerHTML = "<span aria-hidden=\"true\">✦</span> Generate comparison";
    }
  }

  async function loadTraining() {
    try {
      const data = await api("/api/training");
      const stateNode = $("#trainingState");
      const grid = $("#historyGrid");
      if (!data.available) { stateNode.textContent = data.message; grid.replaceChildren(); return; }
      const latest = data.history.at(-1);
      stateNode.textContent = `Loaded ${data.history.length} epochs from the Colab export. The app only reads this record.`;
      grid.replaceChildren(...[
        ["Latest generator loss", latest.generator_loss], ["Latest discriminator loss", latest.discriminator_loss], ["Real-image confidence", latest.discriminator_real], ["Synthetic-image confidence", latest.discriminator_fake]
      ].map(([label, value]) => { const card = document.createElement("article"); card.innerHTML = `<h2>${label}</h2><p class="metric-number">${Number(value).toFixed(4)}</p><p>Recorded by the Colab training run.</p>`; return card; }));
      if (data.samples?.length) {
        const samples = document.createElement("article");
        samples.className = "sample-strip";
        samples.innerHTML = "<h2>Fixed-noise samples</h2><p>Saved at checkpoints, so visual change can be compared honestly.</p>";
        const images = document.createElement("div");
        images.className = "sample-images";
        data.samples.forEach((name) => { const image = document.createElement("img"); image.src = `/api/training-samples/${encodeURIComponent(name)}`; image.alt = `Synthetic sample grid from ${name}`; image.loading = "lazy"; images.append(image); });
        samples.append(images); grid.append(samples);
      }
    } catch { $("#trainingState").textContent = "Training history could not be read from the local service."; }
  }

  async function loadEvaluation() {
    try {
      const data = await api("/api/evaluation");
      const stateNode = $("#evaluationState");
      if (!data.available) { stateNode.textContent = data.message; return; }
      stateNode.replaceChildren();
      Object.entries(data).filter(([key]) => key !== "available").forEach(([key, value]) => {
        const row = document.createElement("p");
        row.textContent = `${key.replaceAll("_", " ")}: ${typeof value === "number" ? value.toFixed(4) : value}`;
        stateNode.append(row);
      });
    } catch { $("#evaluationState").textContent = "Evaluation data could not be read from the local service."; }
  }

  function showView(name) {
    $$(".view").forEach((view) => view.classList.toggle("is-active", view.dataset.view === name));
    $$(".nav-link").forEach((button) => {
      const active = button.dataset.nav === name;
      button.classList.toggle("is-active", active);
      if (active) button.setAttribute("aria-current", "page"); else button.removeAttribute("aria-current");
    });
    if (name === "training") loadTraining();
    if (name === "evaluation") loadEvaluation();
  }

  $$(".count-button").forEach((button) => button.addEventListener("click", () => {
    state.count = Number(button.dataset.count);
    $$(".count-button").forEach((candidate) => { const selected = candidate === button; candidate.classList.toggle("is-selected", selected); candidate.setAttribute("aria-pressed", String(selected)); });
  }));
  $("#randomSeed").addEventListener("click", randomSeed);
  seedInput.addEventListener("blur", validateSeed);
  $("#truncation").addEventListener("input", (event) => { $("#truncationValue").value = Number(event.target.value).toFixed(2); $("#truncationValue").textContent = Number(event.target.value).toFixed(2); });
  generateButton.addEventListener("click", generate);
  $("#copySeed").addEventListener("click", () => { if (state.result) copyText(state.result.seed, "Seed copied. Use it again to reproduce this collection."); });
  $("#downloadGrid").addEventListener("click", () => { if (state.result) download(state.result.grid, state.result.grid_filename); });
  $("#previewDownload").addEventListener("click", () => { if (state.preview) download(state.preview.image, state.preview.filename); });
  $$(".nav-link").forEach((button) => button.addEventListener("click", () => showView(button.dataset.nav)));
  $$("[data-go]").forEach((button) => button.addEventListener("click", () => showView(button.dataset.go)));
  loadModel();
})();
