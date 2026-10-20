(() => {
  const KEY = "cdm-scuola-dualhead-v1";
  const IDS = [
    "c00", "c01", "c02", "c03", "c04", "c05", "c06",
    "c07", "c08", "c09", "c10", "c11", "c12",
  ];

  function load() {
    try {
      return new Set(JSON.parse(localStorage.getItem(KEY) || "[]"));
    } catch {
      return new Set();
    }
  }
  function save(set) {
    localStorage.setItem(KEY, JSON.stringify([...set]));
  }

  const done = load();

  function refreshProgress() {
    const n = IDS.filter((id) => done.has(id)).length;
    const pct = (n / IDS.length) * 100;
    document.getElementById("progress-bar").style.width = pct + "%";
    document.getElementById("progress-label").textContent =
      `${n} / ${IDS.length} capitoli`;
    document.querySelectorAll("#nav a").forEach((a) => {
      const id = a.dataset.id;
      a.classList.toggle("done", done.has(id));
    });
    document.querySelectorAll(".mark-done[data-mark]").forEach((btn) => {
      const id = btn.dataset.mark;
      const isDone = done.has(id);
      btn.classList.toggle("done", isDone);
      btn.textContent = isDone ? "✓ Completato" : btn.dataset.label || btn.textContent.replace(/^✓ Completato$/, "Capitolo completato");
      if (!btn.dataset.label) btn.dataset.label = btn.textContent.startsWith("✓") ? "Capitolo completato" : btn.textContent;
      if (isDone) btn.textContent = "✓ Completato";
      else if (btn.dataset.label) btn.textContent = btn.dataset.label;
    });
  }

  // store original labels once
  document.querySelectorAll(".mark-done[data-mark]").forEach((btn) => {
    btn.dataset.label = btn.textContent;
  });

  document.querySelectorAll(".mark-done[data-mark]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const id = btn.dataset.mark;
      if (done.has(id)) done.delete(id);
      else done.add(id);
      save(done);
      refreshProgress();
    });
  });

  // active nav on scroll
  const sections = IDS.map((id) => document.getElementById(id)).filter(Boolean);
  const navLinks = [...document.querySelectorAll("#nav a")];
  const io = new IntersectionObserver(
    (entries) => {
      entries.forEach((e) => {
        if (!e.isIntersecting) return;
        navLinks.forEach((a) => a.classList.toggle("active", a.getAttribute("href") === "#" + e.target.id));
      });
    },
    { rootMargin: "-20% 0px -60% 0px", threshold: 0.01 }
  );
  sections.forEach((s) => io.observe(s));

  // Quizzes
  document.querySelectorAll(".quiz").forEach((quiz) => {
    const feedback = quiz.querySelector(".feedback");
    quiz.querySelectorAll("button[data-a]").forEach((btn) => {
      btn.addEventListener("click", () => {
        quiz.querySelectorAll("button[data-a]").forEach((b) => {
          b.classList.remove("correct", "wrong");
          b.disabled = true;
        });
        if (btn.hasAttribute("data-ok")) {
          btn.classList.add("correct");
          feedback.textContent = "Giusto. Puoi segnare il capitolo.";
          feedback.style.color = "var(--ok)";
        } else {
          btn.classList.add("wrong");
          const ok = quiz.querySelector("button[data-ok]");
          if (ok) ok.classList.add("correct");
          feedback.textContent = "Non proprio — guarda la risposta evidenziata e rileggi il capitolo.";
          feedback.style.color = "var(--warn)";
        }
      });
    });
  });

  // Lab: prior
  function updPrior() {
    const v = +document.getElementById("lab-vendite").value;
    const h = +document.getElementById("lab-horizon").value;
    document.getElementById("lab-vendite-v").textContent = v;
    document.getElementById("lab-horizon-v").textContent = h;
    const rr = v / 14;
    const prior = rr * h;
    document.getElementById("lab-rr").textContent = rr.toFixed(3);
    document.getElementById("lab-prior").textContent = prior.toFixed(2);
  }
  ["lab-vendite", "lab-horizon"].forEach((id) => {
    document.getElementById(id).addEventListener("input", updPrior);
  });
  updPrior();

  // Lab: neuron
  function updNeuron() {
    const ids = ["n-x1", "n-w1", "n-x2", "n-w2", "n-b"];
    const vals = {};
    ids.forEach((id) => {
      vals[id] = +document.getElementById(id).value;
      document.getElementById(id + "-v").textContent = vals[id].toFixed(1);
    });
    const z = vals["n-w1"] * vals["n-x1"] + vals["n-w2"] * vals["n-x2"] + vals["n-b"];
    const r = Math.max(0, z);
    document.getElementById("n-z").textContent = z.toFixed(2);
    document.getElementById("n-relu").textContent = r.toFixed(2);
    document.getElementById("n-meter").style.width = Math.min(100, (r / 4) * 100) + "%";
    document.getElementById("n-msg").textContent =
      z < 0
        ? "z negativo → ReLU spegne il neurone (0)."
        : "z positivo → il segnale passa.";
  }
  ["n-x1", "n-w1", "n-x2", "n-w2", "n-b"].forEach((id) => {
    document.getElementById(id).addEventListener("input", updNeuron);
  });
  updNeuron();

  // Lab: activations
  function softplus(z) {
    // stable
    if (z > 20) return z;
    return Math.log(1 + Math.exp(z));
  }
  function updAct() {
    const z = +document.getElementById("act-z").value;
    document.getElementById("act-z-v").textContent = z.toFixed(1);
    document.getElementById("act-relu").textContent = Math.max(0, z).toFixed(3);
    document.getElementById("act-sp").textContent = softplus(z).toFixed(3);
  }
  document.getElementById("act-z").addEventListener("input", updAct);
  updAct();

  // Lab: dropout
  const viz = document.getElementById("drop-viz");
  for (let i = 0; i < 24; i++) {
    const s = document.createElement("span");
    viz.appendChild(s);
  }
  function simDrop() {
    const p = +document.getElementById("drop-p").value / 100;
    document.getElementById("drop-p-v").textContent = Math.round(p * 100) + "%";
    let on = 0;
    [...viz.children].forEach((el) => {
      const keep = Math.random() > p;
      el.classList.toggle("on", keep);
      if (keep) on++;
    });
    document.getElementById("drop-msg").textContent =
      `Attivi ${on}/24 in questo passo (dropout ${(p * 100).toFixed(0)}%). In inference sarebbero 24/24.`;
  }
  document.getElementById("drop-p").addEventListener("input", () => {
    document.getElementById("drop-p-v").textContent = document.getElementById("drop-p").value + "%";
  });
  document.getElementById("drop-btn").addEventListener("click", simDrop);
  simDrop();

  // Lab: formula
  function updFormula() {
    const prior = +document.getElementById("f-prior").value;
    const scale = +document.getElementById("f-scale").value;
    const delta = +document.getElementById("f-delta").value;
    document.getElementById("f-prior-v").textContent = prior.toFixed(2);
    document.getElementById("f-scale-v").textContent = scale.toFixed(3);
    document.getElementById("f-delta-v").textContent = (delta >= 0 ? "+" : "") + delta.toFixed(3);
    const mid = prior * scale;
    const y = Math.max(0, mid + delta);
    document.getElementById("f-mid").textContent = mid.toFixed(2);
    document.getElementById("f-out").textContent = y.toFixed(2);
  }
  ["f-prior", "f-scale", "f-delta"].forEach((id) => {
    document.getElementById(id).addEventListener("input", updFormula);
  });
  updFormula();

  // Glossary search
  const search = document.getElementById("gloss-search");
  search.addEventListener("input", () => {
    const q = search.value.trim().toLowerCase();
    document.querySelectorAll("#glossary details").forEach((d) => {
      const hit = !q || d.dataset.g.includes(q) || d.textContent.toLowerCase().includes(q);
      d.style.display = hit ? "" : "none";
    });
  });

  refreshProgress();
})();
