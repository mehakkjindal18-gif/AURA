/* Beyond the Social Spectrum: frontend logic. All predictions and statistics come from the Flask API. */
const $ = id => document.getElementById(id);
const pct = v => (v * 100).toFixed(1) + "%", n3 = v => v.toFixed(3);
const LABEL = {Time_spent_Alone:"Time spent alone", Stage_fear:"Stage fear", Social_event_attendance:"Social event attendance",
  Going_outside:"Going outside", Drained_after_socializing:"Drained after socializing", Friends_circle_size:"Friends circle size", Post_frequency:"Post frequency"};

/* ---------- Data request shared by the loader and the dashboard ---------- */
const metricsReq = fetch("/api/metrics").then(r => { if (!r.ok) throw new Error("metrics"); return r.json(); });

/* ---------- Questionnaire (same feature names and ranges as the Flask backend) ---------- */
const Q = [
  {k:"Time_spent_Alone", t:"How many hours do you spend alone per day?", min:0, max:11, hint:"Hours per day, from 0 to 11."},
  {k:"Stage_fear", t:"Do you experience stage fear?", yn:1, hint:"Choose one answer."},
  {k:"Social_event_attendance", t:"How many social events do you attend?", min:0, max:10, hint:"On a scale from 0 to 10."},
  {k:"Going_outside", t:"How often do you go outside?", min:0, max:7, hint:"On a scale from 0 to 7."},
  {k:"Drained_after_socializing", t:"Do you feel drained after socializing?", yn:1, hint:"Choose one answer."},
  {k:"Friends_circle_size", t:"How big is your close friends circle?", min:0, max:15, hint:"Number of close friends, from 0 to 15."},
  {k:"Post_frequency", t:"How often do you post on social media?", min:0, max:10, hint:"On a scale from 0 to 10."},
];
const quiz = $("quiz");
let i = 0, ans = {}, error = "";

function draw() {
  const q = Q[i], v = ans[q.k], last = i === Q.length - 1;
  const ctl = q.yn
    ? `<div class="yn">${["Yes","No"].map(o => `<button class="opt ${v===o?"on":""}" data-v="${o}" aria-pressed="${v===o}">${o}</button>`).join("")}</div>`
    : `<div class="val" id="val">${v ?? "–"}</div>
       <input type="range" min="${q.min}" max="${q.max}" step="1" value="${v ?? Math.round((q.min + q.max) / 2)}" aria-label="${q.t}">
       <div class="scale"><span>${q.min}</span><span>${q.max}</span></div>
       ${v === undefined ? '<p class="note small" id="slidehint" style="margin-top:1rem">Move the slider to set your answer.</p>' : ""}`;
  quiz.innerHTML = `<div class="bar" role="progressbar" aria-valuemin="1" aria-valuemax="${Q.length}" aria-valuenow="${i+1}"><i style="width:${(i+1)/Q.length*100}%"></i></div>
    <div class="step">Question ${i+1} of ${Q.length}</div><h3 class="q">${q.t}</h3><p class="hint">${q.hint}</p><div class="body">${ctl}</div>
    ${error ? `<div class="err" role="alert">${error}</div>` : ""}
    <div class="navbtns"><button class="pill outline" id="back" ${i===0?"disabled":""}>Back</button>
    <button class="pill solid" id="next" ${v===undefined?"disabled":""}>${last?"Get my prediction":"Continue"}</button></div>`;
  quiz.querySelectorAll(".opt").forEach(b => b.onclick = () => { ans[q.k] = b.dataset.v; error = ""; draw(); });
  const r = quiz.querySelector("input[type=range]");
  if (r) r.oninput = () => {
    ans[q.k] = +r.value; $("val").textContent = r.value;
    $("next").disabled = false; $("slidehint")?.remove();
  };
  $("back").onclick = () => { i--; error = ""; draw(); };
  $("next").onclick = () => last ? submit() : (i++, error = "", draw());
}

async function submit() {
  quiz.innerHTML = `<div class="load"><div class="spin"></div>Running the model…</div>`;
  try {
    const res = await fetch("/api/predict", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(ans)});
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.error || "The server could not process this request.");
    result(data);
  } catch (e) {
    error = e instanceof TypeError ? "Could not reach the server. Check that the Flask app is running, then try again." : e.message;
    draw();
  }
}

function result(d) {
  const p = d.extrovert_probability, C = 2 * Math.PI * 52, ex = d.prediction === "Extrovert";
  quiz.innerHTML = `<div class="res">
    <svg class="ring" viewBox="0 0 120 120" role="img" aria-label="Extrovert probability ${pct(p)}">
      <circle cx="60" cy="60" r="52" fill="none" stroke="#e2e8f0" stroke-width="10"/>
      <circle cx="60" cy="60" r="52" fill="none" stroke="${ex ? "#0d9488" : "#6366f1"}" stroke-width="10" stroke-linecap="round" stroke-dasharray="${C*p} ${C}" transform="rotate(-90 60 60)"/>
      <text x="60" y="62" text-anchor="middle" style="font-size:22px">${pct(p)}</text><text x="60" y="76" text-anchor="middle" class="s">Extrovert probability</text></svg>
    <h3>Predicted: ${d.prediction}</h3>
    <div class="rspec"><div class="track"><i id="rmark" style="--p:${p*100}%"></i></div><div class="ends"><span>Introvert</span><span>Extrovert</span></div></div>
    <p>The model estimates a <b>${pct(p)}</b> probability that these answers belong to the Extrovert class, and ${pct(d.introvert_probability)} for Introvert. The label is Extrovert when the Extrovert probability is 50% or higher.</p>
    <p>This is the model's estimate for the Extrovert class based on seven answers. It is not a measurement of your whole personality.</p>
    <button class="pill solid" id="again">Retake the questionnaire</button></div>`;
  requestAnimationFrame(() => requestAnimationFrame(() => { $("rmark").style.left = (p * 100) + "%"; }));
  $("again").onclick = () => { ans = {}; i = 0; error = ""; draw(); };
}
draw();

/* ---------- Dashboard (values come from /api/metrics) ---------- */
const DEFS = [["accuracy","Accuracy","Share of correct predictions",pct],["precision","Precision","Predicted extroverts that were right",pct],
  ["recall","Recall","Actual extroverts the model found",pct],["f1","F1 score","Balance of precision and recall",pct],
  ["roc_auc","ROC-AUC","How well the classes are ranked",n3],["mcc","MCC","Matthews correlation coefficient",n3],["kappa","Cohen's Kappa","Agreement beyond chance",n3]];

function renderDashboard(M) {
  const rf = M.test["Random Forest"], lr = M.test["Logistic Regression"];
  $("hero-acc-v").textContent = pct(rf.accuracy); $("hero-acc").hidden = false;
  $("perf-note").textContent = `Hold-out test set of ${M.n_test} respondents (20% split). Metrics below are for the Random Forest unless stated.`;
  $("metrics").innerHTML = DEFS.map(([k,l,d,f]) => `<div><b>${f(rf[k])}</b><span>${l}</span><small>${d}</small></div>`).join("");

  const [[tn,fp],[fn,tp]] = rf.confusion_matrix, mx = Math.max(tn,fp,fn,tp);
  const cell = (v, ok) => `<div class="c" style="background:rgba(${ok?"99,102,241":"13,148,136"},${.1+.5*v/mx})">${v}</div>`;
  $("cm").innerHTML = `<div class="cm"><span></span><span>Predicted Introvert</span><span>Predicted Extrovert</span>
    <span>Actual Introvert</span>${cell(tn,1)}${cell(fp,0)}<span>Actual Extrovert</span>${cell(fn,0)}${cell(tp,1)}</div>`;

  const X = v => 34 + v * 176, Y = v => 186 - v * 176, pts = m => m.roc.fpr.map((x,k) => `${X(x).toFixed(1)},${Y(m.roc.tpr[k]).toFixed(1)}`).join(" ");
  $("roc").innerHTML = `<svg viewBox="0 0 230 222" role="img" aria-label="ROC curves for Random Forest and Logistic Regression" style="width:100%;height:auto;max-width:26rem">
    <rect x="34" y="10" width="176" height="176" fill="none" stroke="#e2e8f0"/><line x1="34" y1="186" x2="210" y2="10" stroke="#aab" stroke-dasharray="4"/>
    <polyline points="${pts(lr)}" fill="none" stroke="#0d9488" stroke-width="2"/><polyline points="${pts(rf)}" fill="none" stroke="#6366f1" stroke-width="2.5"/>
    ${[0,.5,1].map(t => `<text x="${X(t)}" y="198" text-anchor="middle" font-size="8" fill="#5f6675">${t}</text><text x="29" y="${Y(t)+3}" text-anchor="end" font-size="8" fill="#5f6675">${t}</text>`).join("")}
    <text x="122" y="216" text-anchor="middle" font-size="8" fill="#5f6675">False positive rate</text>
    <text transform="translate(8 98) rotate(-90)" text-anchor="middle" font-size="8" fill="#5f6675">True positive rate</text></svg>
    <div class="legend"><span><i style="background:#6366f1"></i>Random Forest (AUC ${n3(rf.roc_auc)})</span><span><i style="background:#0d9488"></i>Logistic Regression (AUC ${n3(lr.roc_auc)})</span></div>`;

  const head = ks => `<tr><th>Model</th>${ks.map(k => `<th>${k[1]}</th>`).join("")}</tr>`;
  $("cmp-test").innerHTML = `<table>${head(DEFS)}${[["Random Forest",rf],["Logistic Regression",lr]].map(([n,m]) =>
    `<tr class="${n==="Random Forest"?"hl":""}"><td>${n}</td>${DEFS.map(([k,,,f]) => `<td>${f(m[k])}</td>`).join("")}</tr>`).join("")}</table>`;

  const K = [["accuracy","Accuracy"],["auc","AUC"],["recall","Recall"],["precision","Precision"],["f1","F1"],["kappa","Kappa"],["mcc","MCC"]];
  const skipped = ["Quadratic Discriminant Analysis"].filter(n => !M.cv_comparison.some(r => r.model === n));
  $("cmp-note").textContent = "Mean of 10-fold stratified cross-validation on the full dataset, sorted by accuracy. Highlighted rows are the two focus models." +
    (skipped.length ? ` ${skipped.join(", ")} could not be fitted on this data and is not shown.` : "");
  $("cmp").innerHTML = `<table>${head(K)}${M.cv_comparison.map(r => `<tr class="${/^(Random Forest|Logistic Regression)$/.test(r.model)?"hl":""}"><td>${r.model}</td>${K.map(k => `<td>${n3(r[k[0]])}</td>`).join("")}</tr>`).join("")}</table>`;

  const top = Math.max(...M.feature_importance.map(f => f.importance));
  $("fi").innerHTML = M.feature_importance.map(f => `<div class="fi"><span>${LABEL[f.feature] || f.feature}</span><span><i style="width:${f.importance/top*100}%"></i></span><span>${f.importance.toFixed(3)}</span></div>`).join("");
}
metricsReq.then(renderDashboard).catch(() => { $("perf-note").textContent = "Results are unavailable. Run train_export.py, then restart the Flask app."; });

/* ---------- Page shell: loader, hero reveal, parallax, menu, smooth scroll ---------- */
let lenis = null;
import("https://cdn.jsdelivr.net/npm/lenis@1.1.18/+esm").then(m => {   // optional enhancement
  lenis = new m.default({smoothWheel: true, anchors: true});
  const raf = t => { lenis.raf(t); requestAnimationFrame(raf); }; requestAnimationFrame(raf);
}).catch(() => {});

let pr = 0;
const tick = setInterval(() => { pr += (90 - pr) * .12; $("loader-fill").style.width = pr + "%"; $("loader-counter").textContent = Math.round(pr) + "%"; }, 60);
Promise.allSettled([Promise.race([metricsReq, new Promise(r => setTimeout(r, 4000))]), new Promise(r => setTimeout(r, 900))]).then(() => {
  clearInterval(tick); $("loader-fill").style.width = "100%"; $("loader-counter").textContent = "100%";
  setTimeout(() => { $("loader").classList.add("loaded"); document.querySelectorAll(".hero-title .clip-wrap").forEach(e => e.classList.add("revealed")); }, 300);
});

const heroImg = $("hero-img"), fab = $("fab");
addEventListener("scroll", () => {
  if (heroImg && scrollY <= innerHeight) heroImg.style.transform = `translateY(${scrollY * .15}px)`;
  fab.classList.toggle("show", scrollY > innerHeight * .7);
}, {passive: true});

const menu = $("menu");
window.toggleMenu = () => {
  const on = menu.classList.toggle("active"); menu.setAttribute("aria-hidden", String(!on));
  if (lenis) on ? lenis.stop() : lenis.start();
};
addEventListener("keydown", e => { if (e.key === "Escape" && menu.classList.contains("active")) toggleMenu(); });