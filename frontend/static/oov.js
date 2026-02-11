// frontend/static/oov.js
async function getJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(await res.text());
  return await res.json();
}

const out = document.getElementById("out");
const limitEl = document.getElementById("limit");

async function loadRejected() {
  const limit = Number(limitEl.value || 200);
  const data = await getJSON(`/ui/rejected_candidates?limit=${limit}`);
  out.textContent = JSON.stringify(data, null, 2);
}

async function loadConfirmed() {
  const data = await getJSON(`/ui/oov_candidates`);
  out.textContent = JSON.stringify(data, null, 2);
}

document.getElementById("btnRejected").onclick = () => loadRejected().catch(e => out.textContent = String(e));
document.getElementById("btnConfirmed").onclick = () => loadConfirmed().catch(e => out.textContent = String(e));

loadRejected().catch(e => out.textContent = String(e));
