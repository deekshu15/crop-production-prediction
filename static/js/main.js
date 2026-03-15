window.addEventListener('load', () => {
  const loader = document.getElementById('loader');
  if (loader) {
    loader.classList.add('loader-hidden');
  }
});

function showToast(message, type = 'success') {
  const toast = document.createElement('div');
  toast.className = `toast ${type === 'error' ? 'error' : ''}`;
  toast.textContent = message;
  document.body.appendChild(toast);
  setTimeout(() => toast.remove(), 3000);
}

async function loadStats() {
  try {
    const res = await fetch('/api/stats');
    if (!res.ok) throw new Error('Failed to fetch stats');

    const stats = await res.json();

    const stateSelect = document.getElementById('state-select');
    const cropSelect = document.getElementById('crop-select');
    const seasonSelect = document.getElementById('season-select');

    if (stateSelect) {
      stateSelect.innerHTML = '<option value="">Select State</option>';
      stats.states_list.forEach((state) => {
        const option = document.createElement('option');
        option.value = state;
        option.textContent = state;
        stateSelect.appendChild(option);
      });
    }

    if (cropSelect) {
      cropSelect.innerHTML = '<option value="">Select Crop</option>';
      stats.crops_list.forEach((crop) => {
        const option = document.createElement('option');
        option.value = crop;
        option.textContent = crop;
        cropSelect.appendChild(option);
      });
    }

    if (seasonSelect) {
      seasonSelect.innerHTML = '<option value="">Select Season</option>';
      stats.seasons_list.forEach((season) => {
        const option = document.createElement('option');
        option.value = season;
        option.textContent = season;
        seasonSelect.appendChild(option);
      });
    }

    const recordsCounter = document.getElementById('records-counter');
    const cropsCounter = document.getElementById('crops-counter');
    const statesCounter = document.getElementById('states-counter');
    const yearsCounter = document.getElementById('years-counter');

    if (recordsCounter) recordsCounter.textContent = `${stats.total_records.toLocaleString()}+`;
    if (cropsCounter) cropsCounter.textContent = `${stats.crops_list.length}+`;
    if (statesCounter) statesCounter.textContent = String(stats.states_list.length);
    if (yearsCounter) yearsCounter.textContent = String(stats.year_range.max - stats.year_range.min);

    if (document.body.dataset.page === 'index') {
      initFeatureChart(stats.feature_importance);
    }

  } catch (error) {
    showToast('Failed to load data', 'error');
  }
}

function initStateDistrictHandler() {
  const stateSelect = document.getElementById('state-select');
  if (!stateSelect) return;

  stateSelect.addEventListener('change', async (e) => {
    const state = e.target.value;
    const districtSelect = document.getElementById('district-select');

    if (!districtSelect) return;

    if (!state) {
      districtSelect.innerHTML = '<option value="">Select State First</option>';
      return;
    }

    districtSelect.innerHTML = '<option value="">Loading...</option>';

    try {
      const res = await fetch(`/api/districts/${encodeURIComponent(state)}`);
      const data = await res.json();
      districtSelect.innerHTML = '<option value="">Select District</option>';

      data.districts.forEach((d) => {
        const opt = document.createElement('option');
        opt.value = d;
        opt.textContent = d;
        districtSelect.appendChild(opt);
      });
    } catch (error) {
      districtSelect.innerHTML = '<option value="">Select State First</option>';
      showToast('Unable to load districts', 'error');
    }
  });
}

function setFieldError(field, message) {
  field.classList.add('error');
  const errorEl = field.parentElement.querySelector('.error-text');
  if (errorEl) errorEl.textContent = message;
}

function clearFieldError(field) {
  field.classList.remove('error');
  const errorEl = field.parentElement.querySelector('.error-text');
  if (errorEl) errorEl.textContent = '';
}

function validateForm() {
  const fields = [
    { id: 'state-select', message: 'Please select a state' },
    { id: 'district-select', message: 'Please select a district' },
    { id: 'season-select', message: 'Please select a season' },
    { id: 'crop-select', message: 'Please select a crop' },
    { id: 'year-input', message: 'Please enter a valid year' },
    { id: 'area-input', message: 'Please enter area greater than 0' },
  ];

  let valid = true;

  fields.forEach(({ id, message }) => {
    const field = document.getElementById(id);
    if (!field) return;

    clearFieldError(field);

    const value = field.value.trim();
    if (!value) {
      setFieldError(field, message);
      valid = false;
      return;
    }

    if (id === 'area-input' && parseFloat(value) <= 0) {
      setFieldError(field, message);
      valid = false;
    }
  });

  return valid;
}

function initFormValidationReset() {
  const inputs = document.querySelectorAll('#predict-form input, #predict-form select');
  inputs.forEach((input) => {
    input.addEventListener('focus', () => clearFieldError(input));
  });
}

function displayResult(data) {
  const card = document.getElementById('result-card');
  if (!card) return;

  card.classList.add('show');
  card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

  document.getElementById('result-production').textContent = `${Math.round(data.prediction).toLocaleString()} t`;
  document.getElementById('result-yield').textContent = `${data.yield_per_hectare.toFixed(2)} t/ha`;
  document.getElementById('result-interpretation').textContent = data.interpretation;
  document.getElementById('result-model').textContent = data.model_used;

  const bar = document.getElementById('confidence-fill');
  bar.style.width = '0%';
  setTimeout(() => {
    bar.style.width = `${data.confidence}%`;
  }, 100);
  document.getElementById('confidence-label').textContent = `${data.confidence}%`;
}

function initPredictionForm() {
  const form = document.getElementById('predict-form');
  if (!form) return;

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    if (!validateForm()) return;

    const btn = document.getElementById('predict-btn');
    btn.innerHTML = '<span class="spinner"></span> Predicting...';
    btn.disabled = true;

    const payload = {
      state: document.getElementById('state-select').value,
      district: document.getElementById('district-select').value,
      year: parseInt(document.getElementById('year-input').value, 10),
      season: document.getElementById('season-select').value,
      crop: document.getElementById('crop-select').value,
      area: parseFloat(document.getElementById('area-input').value),
    };

    try {
      const res = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Prediction failed');
      displayResult(data);
    } catch (err) {
      showToast(err.message, 'error');
    } finally {
      btn.innerHTML = 'Predict Yield →';
      btn.disabled = false;
    }
  });

  const resetBtn = document.getElementById('new-prediction-btn');
  if (resetBtn) {
    resetBtn.addEventListener('click', () => {
      form.reset();
      document.getElementById('result-card').classList.remove('show');
      document.getElementById('district-select').innerHTML = '<option value="">Select State First</option>';
    });
  }
}

function initFeatureChart(importanceData) {
  const canvas = document.getElementById('featureChart');
  if (!canvas || !importanceData) return;

  const labels = Object.keys(importanceData);
  const values = Object.values(importanceData);

  new Chart(canvas, {
    type: 'bar',
    data: {
      labels,
      datasets: [
        {
          data: values,
          backgroundColor: values.map((v, i) => `rgba(57,255,20,${0.4 + (0.6 * i) / values.length})`),
          borderColor: '#39FF14',
          borderWidth: 1,
          borderRadius: 4,
        },
      ],
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: '#1a1a1a',
          borderColor: '#39FF14',
          borderWidth: 1,
          titleColor: '#ffffff',
          bodyColor: '#ffffff',
        },
      },
      scales: {
        x: {
          grid: { color: 'rgba(255,255,255,0.05)' },
          ticks: { color: '#aaaaaa' },
        },
        y: {
          grid: { color: 'rgba(255,255,255,0.05)' },
          ticks: { color: '#aaaaaa' },
        },
      },
    },
  });
}

function initNavbarScrollGlow() {
  const navbar = document.getElementById('navbar');
  if (!navbar) return;

  window.addEventListener('scroll', () => {
    if (window.scrollY > 50) {
      navbar.classList.add('scrolled');
    } else {
      navbar.classList.remove('scrolled');
    }
  });
}

function initHeroButtonScroll() {
  const btn = document.getElementById('start-btn');
  if (!btn) return;

  btn.addEventListener('click', () => {
    const section = document.getElementById('predict-section');
    if (section) section.scrollIntoView({ behavior: 'auto' });
  });
}

document.addEventListener('DOMContentLoaded', () => {
  initHeroButtonScroll();

  if (document.getElementById('predict-form')) {
    initStateDistrictHandler();
    initFormValidationReset();
    initPredictionForm();
  }

  loadStats();
});
