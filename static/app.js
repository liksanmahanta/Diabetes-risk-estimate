const form = document.querySelector("#prediction-form");
const submitButton = document.querySelector("#submit-button");
const formError = document.querySelector("#form-error");
const modelStatus = document.querySelector("#model-status");
const modelStatusText = document.querySelector("#model-status-text");
const resultPanel = document.querySelector(".result-panel");
const resultEmpty = document.querySelector("#result-empty");
const resultContent = document.querySelector("#result-content");

function updateResultSectionVisibility() {
  const requiredFields = form.querySelectorAll("input[required], select[required]");
  const allRequiredFilled = Array.from(requiredFields).every((field) => {
    if (field.tagName === "SELECT") {
      return field.value !== "";
    }
    return field.value.trim() !== "";
  });

  if (!allRequiredFilled) {
    resultPanel.hidden = true;
    return;
  }

  resultPanel.hidden = false;
  resultEmpty.hidden = false;
  resultContent.hidden = true;
  document.querySelector("#result-empty p").textContent = "All required fields are complete. Press Estimate risk to open the model output.";
}

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
    updateResultSectionVisibility();
  } catch (error) {
    showModelUnavailable(error.message);
    document.querySelector("#selected-model-chip").textContent = "Train required";
    resultPanel.hidden = true;
  }
}

let patientRecords = [];

function renderPatientRecords() {
  const body = document.querySelector("#patient-records-body");
  const status = document.querySelector("#records-status");
  const query = document.querySelector("#record-search").value.trim().toLocaleLowerCase();
  const matchingRecords = patientRecords.filter((record) => record.patient_name.toLocaleLowerCase().includes(query));
  body.replaceChildren();
  if (matchingRecords.length === 0) {
    const row = document.createElement("tr");
    const cell = document.createElement("td");
    cell.colSpan = 8;
    cell.className = "table-empty";
    cell.textContent = query ? "No patient records match that name." : "No patient checkups have been saved yet.";
    row.append(cell);
    body.append(row);
  } else {
    for (const record of matchingRecords) {
      const row = document.createElement("tr");
      const date = new Date(record.created_at);
      const values = [
        record.patient_name,
        Number.isNaN(date.getTime()) ? record.created_at : date.toLocaleString(),
        record.gender,
        record.age ?? "-",
        record.glucose ?? "-",
        record.bmi ?? "-",
        record.prediction === 1 || record.prediction === "1" ? "Above threshold" : "Below threshold",
        `${(Number(record.probability) * 100).toFixed(1)}%`,
      ];
      for (const value of values) {
        const cell = document.createElement("td");
        cell.textContent = value;
        row.append(cell);
      }
      body.append(row);
    }
  }
  status.textContent = query
    ? `Showing ${matchingRecords.length} of ${patientRecords.length} saved checkups.`
    : `${patientRecords.length} saved checkup${patientRecords.length === 1 ? "" : "s"}.`;
}

async function loadPatientRecords() {
  const body = document.querySelector("#patient-records-body");
  const status = document.querySelector("#records-status");
  if (!body || !status) return;
  status.textContent = "Loading saved records...";
  try {
    const response = await fetch("/api/patient-records");
    const records = await response.json();
    if (!response.ok) {
      if (records.login_url) {
        status.textContent = records.error;
        const link = document.createElement("a");
        link.href = records.login_url;
        link.textContent = " Owner sign in";
        status.append(link);
        return;
      }
      throw new Error(records.error || "Could not load patient records.");
    }
    patientRecords = records;
    renderPatientRecords();
  } catch (error) {
    body.replaceChildren();
    const row = document.createElement("tr");
    const cell = document.createElement("td");
    cell.colSpan = 8;
    cell.className = "table-empty";
    cell.textContent = error.message;
    row.append(cell);
    body.append(row);
    status.textContent = "Records could not be loaded.";
  }
}

document.querySelector("#refresh-records")?.addEventListener("click", loadPatientRecords);
document.querySelector("#record-search")?.addEventListener("input", renderPatientRecords);

form.querySelectorAll("input, select").forEach((field) => {
  field.addEventListener("input", updateResultSectionVisibility);
  field.addEventListener("change", updateResultSectionVisibility);
});

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
    resultPanel.hidden = false;
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
    document.querySelector("#record-save-status").textContent = `Patient record saved (ID ${result.record_id}) in patient_records.db.`;
    await loadPatientRecords();
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
loadPatientRecords();
