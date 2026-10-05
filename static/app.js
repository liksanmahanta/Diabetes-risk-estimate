const form = document.querySelector("#prediction-form");
const submitButton = document.querySelector("#submit-button");
const formError = document.querySelector("#form-error");
const modelStatus = document.querySelector("#model-status");
const modelStatusText = document.querySelector("#model-status-text");

function showModelUnavailable(message) {
  modelStatus.classList.add("error");
  modelStatusText.textContent = "Model not ready";
  document.querySelector("#metrics-body").innerHTML = `<tr><td colspan="5" class="table-empty">${message}</td></tr>`;
  document.querySelector("#importance-list").innerHTML = `<li class="table-empty">${message}</li>`;
  document.querySelector("#evaluation-caption").textContent = "Train the model first";
}

function renderMetrics(info) {
  const body = document.querySelector("#metrics-body");
  body.replaceChildren();
  for (const [name, scores] of Object.entries(info.metrics)) {
    const row = document.createElement("tr");
    if (name === info.selected_model) row.classList.add("selected");
    const values = [name, scores.accuracy, scores.precision, scores.recall, scores.f1];
    for (const value of values) {
      const cell = document.createElement("td");
      cell.textContent = typeof value === "number" ? value.toFixed(3) : value;
      row.append(cell);
    }
    body.append(row);
  }
  document.querySelector("#evaluation-caption").textContent = `${info.test_rows} holdout / selected by 5-fold CV`;

  const list = document.querySelector("#importance-list");
  list.replaceChildren();
  const maximum = Math.max(...info.feature_importance.map((item) => item.importance), 0.00001);
  for (const item of info.feature_importance) {
    const row = document.createElement("li");
    row.className = "importance-item";
    const label = document.createElement("span");
    label.textContent = item.label;
    const track = document.createElement("span");
    track.className = "importance-track";
    const fill = document.createElement("span");
    fill.className = "importance-fill";
    fill.style.width = `${Math.max(0, item.importance / maximum * 100)}%`;
    track.append(fill);
    const value = document.createElement("strong");
    value.textContent = item.importance.toFixed(3);
    row.append(label, track, value);
    list.append(row);
  }
}

async function loadModelInfo() {
  try {
    const response = await fetch("/api/model-info");
    const info = await response.json();
    if (!response.ok) throw new Error(info.error || "Model information could not be loaded.");
    renderMetrics(info);
    modelStatus.classList.add("ready");
    modelStatusText.textContent = "Model ready";
    document.querySelector("#selected-model-chip").textContent = info.selected_model;
    submitButton.disabled = false;
  } catch (error) {
    showModelUnavailable(error.message);
    document.querySelector("#selected-model-chip").textContent = "Train required";
  }
}

function renderExplanation(items) {
  const list = document.querySelector("#sensitivity-list");
  list.replaceChildren();
  const maximum = Math.max(...items.map((item) => Math.abs(item.delta_points)), 1);
  for (const item of items.slice(0, 5)) {
    const row = document.createElement("li");
    row.className = "sensitivity-item";
    const label = document.createElement("span");
    label.className = "sensitivity-label";
    label.textContent = item.label;
    const track = document.createElement("span");
    track.className = "sensitivity-track";
    const fill = document.createElement("span");
    fill.className = `sensitivity-fill${item.delta_points < 0 ? " negative" : ""}`;
    fill.style.width = `${Math.abs(item.delta_points) / maximum * 100}%`;
    track.append(fill);
    const value = document.createElement("span");
    value.className = "sensitivity-value";
    value.textContent = `${item.delta_points > 0 ? "+" : ""}${item.delta_points.toFixed(1)} pts`;
    row.append(label, track, value);
    list.append(row);
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  formError.textContent = "";
  submitButton.disabled = true;
  submitButton.querySelector("span:first-child").textContent = "Estimating...";
  const values = Object.fromEntries(new FormData(form).entries());
  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(values),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Prediction failed.");
    const percent = result.probability * 100;
    document.querySelector("#result-empty").hidden = true;
    document.querySelector("#result-content").hidden = false;
    document.querySelector("#score-value").textContent = `${percent.toFixed(1)}%`;
    document.querySelector("#score-fill").style.width = `${percent}%`;
    document.querySelector("#score-track").setAttribute("aria-label", `Estimated probability ${percent.toFixed(1)} percent`);
    const classification = document.querySelector("#classification");
    classification.textContent = result.prediction ? "Above model threshold" : "Below model threshold";
    classification.classList.toggle("positive", result.prediction === 1);
    document.querySelector("#result-caveat").textContent = result.missing_measurements.length
      ? `Missing ${result.missing_measurements.map((name) => name.replace(/([A-Z])/g, " $1").trim().toLowerCase()).join(", ")} was imputed. Probability is not calibrated for clinical use.`
      : "Probability is not calibrated for clinical use. The 0.50 classification threshold is a prototype setting, not a clinical cutoff.";
    document.querySelector("#selected-model-chip").textContent = result.selected_model;
    renderExplanation(result.explanation);
  } catch (error) {
    formError.textContent = error.message;
  } finally {
    submitButton.disabled = false;
    submitButton.querySelector("span:first-child").textContent = "Estimate risk";
  }
});

loadModelInfo();