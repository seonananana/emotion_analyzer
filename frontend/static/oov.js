// frontend/static/oov.js

async function getJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(await res.text());
  return await res.json();
}

async function postJSON(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body ?? {}),
  });
  if (!res.ok) throw new Error(await res.text());
  return await res.json();
}

const out = document.getElementById("out");
const limitEl = document.getElementById("limit");
const includeProcessedEl = document.getElementById("includeProcessed");

const quickTextEl = document.getElementById("quickText");
const quickOutEl = document.getElementById("quickOut");
const quickThresholdEl = document.getElementById("quickThreshold");
const btnRunEl = document.getElementById("btnRun");

function escapeHtml(s) {
  return String(s ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function renderError(e) {
  out.innerHTML = `<pre style="white-space:pre-wrap;color:#b91c1c;">${escapeHtml(
    String(e)
  )}</pre>`;
}

function pill(status) {
  const s = String(status || "pending");
  const base =
    "display:inline-flex;align-items:center;gap:6px;padding:2px 8px;border-radius:999px;font-size:12px;font-weight:700;border:1px solid;";
  if (s === "approved")
    return `<span style="${base} border-color:#16a34a;color:#166534;background:#dcfce7;">✅ approved</span>`;
  if (s === "rejected")
    return `<span style="${base} border-color:#ef4444;color:#7f1d1d;background:#fee2e2;">⛔ rejected</span>`;
  return `<span style="${base} border-color:#64748b;color:#334155;background:#f1f5f9;">⏳ pending</span>`;
}

/**
 * ✅ routes_oov.py 기준 API
 * - GET  /ui/oov_candidates  (기본 include_review=true) -> records[*].review_status / review_notes / review_updated_at
 * - POST /ui/oov_candidates/approve  body { key, notes }
 * - POST /ui/oov_candidates/reject   body { key, notes }
 * - POST /ui/oov_candidates/pending  body { key, notes }
 */

async function setStatus(key, status, notes) {
  const url =
    status === "approved"
      ? "/ui/oov_candidates/approve"
      : status === "rejected"
      ? "/ui/oov_candidates/reject"
      : "/ui/oov_candidates/pending";
  return await postJSON(url, { key, notes: notes || "" });
}

// ----------------------
// Confirmed(OOV candidates) view
// ----------------------

function renderConfirmed(records) {
  if (!records || records.length === 0) {
    out.innerHTML = `<div style="padding:12px;">(records 없음)</div>`;
    return;
  }

  out.innerHTML = `
    <div style="display:flex; gap:10px; align-items:center; padding:8px 0; flex-wrap:wrap;">
      <div style="font-weight:800;">OOV Candidates</div>

      <label style="display:flex; gap:6px; align-items:center; font-size:12px; opacity:.85;">
        Status
        <select id="statusFilter" style="padding:6px 8px; border:1px solid #d1d5db; border-radius:8px;">
          <option value="all">all</option>
          <option value="pending" selected>pending</option>
          <option value="approved">approved</option>
          <option value="rejected">rejected</option>
        </select>
      </label>

      <label style="display:flex; gap:6px; align-items:center; font-size:12px; opacity:.85;">
        contains
        <input id="containsFilter" placeholder="예: 못하 / 미치 / 현타"
          style="padding:6px 8px; border:1px solid #d1d5db; border-radius:8px; min-width:240px;"
        />
      </label>

      <button id="btnRefreshConfirmed"
        style="margin-left:auto; padding:6px 10px; border-radius:8px; border:1px solid #0ea5e9; background:#0ea5e9; color:white; cursor:pointer;">
        Refresh
      </button>
    </div>

    <div id="confirmedList" style="display:flex; flex-direction:column; gap:10px;"></div>
  `;

  const listEl = document.getElementById("confirmedList");
  const statusEl = document.getElementById("statusFilter");
  const containsEl = document.getElementById("containsFilter");
  const refreshBtn = document.getElementById("btnRefreshConfirmed");

  function applyFilter() {
    const st = statusEl?.value || "pending";
    const q = (containsEl?.value || "").trim();
    const q2 = q.replace(/\s+/g, "").toLowerCase();

    const filtered = records.filter((r) => {
      const rs = String(r.review_status || "pending");
      if (st !== "all" && rs !== st) return false;
      if (!q) return true;
      const k = String(r.key || r.text || "");
      const k2 = k.replace(/\s+/g, "").toLowerCase();
      return k2.includes(q2);
    });

    listEl.innerHTML = "";
    for (const r of filtered) {
      const key = r.key || r.text || "";
      const cnt = r.count ?? 0;
      const last = r.last_seen_at || "";
      const ex = Array.isArray(r.examples) ? r.examples.slice(0, 3) : [];
      const st2 = r.review_status || "pending";
      const notes = r.review_notes || "";

      const card = document.createElement("div");
      card.style.border = "1px solid #e5e7eb";
      card.style.borderRadius = "10px";
      card.style.padding = "10px";

      card.innerHTML = `
        <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:10px;">
          <div>
            <div style="font-weight:800; font-size:16px;">
              ${escapeHtml(key)}
              <span style="margin-left:8px;">${pill(st2)}</span>
            </div>
            <div style="margin-top:4px; font-size:12px; opacity:.75;">
              count=${escapeHtml(cnt)} ${last ? `· last=${escapeHtml(last)}` : ""}
            </div>
          </div>

          <div style="display:flex; gap:6px; flex-wrap:wrap; align-items:center;">
            <button data-set="approved" data-key="${escapeHtml(key)}"
              style="padding:6px 10px; border-radius:8px; border:1px solid #16a34a; background:#16a34a; color:white; cursor:pointer;">
              Approve
            </button>
            <button data-set="rejected" data-key="${escapeHtml(key)}"
              style="padding:6px 10px; border-radius:8px; border:1px solid #ef4444; background:#ef4444; color:white; cursor:pointer;">
              Reject
            </button>
            <button data-set="pending" data-key="${escapeHtml(key)}"
              style="padding:6px 10px; border-radius:8px; border:1px solid #64748b; background:#f1f5f9; color:#334155; cursor:pointer;">
              Pending
            </button>
          </div>
        </div>

        <div style="margin-top:8px; display:flex; gap:10px; flex-wrap:wrap; align-items:center;">
          <input data-notes="${escapeHtml(key)}"
            placeholder="notes (선택)"
            value="${escapeHtml(notes)}"
            style="padding:6px 8px; border:1px solid #d1d5db; border-radius:8px; min-width:320px; flex:1;"
          />
        </div>

        ${
          ex.length
            ? `<details style="margin-top:8px;">
                 <summary style="cursor:pointer; opacity:.8;">examples</summary>
                 <ul style="margin:6px 0 0 18px; padding:0; font-size:12px; opacity:.9;">
                   ${ex.map((t) => `<li style="margin:4px 0;">${escapeHtml(t)}</li>`).join("")}
                 </ul>
               </details>`
            : ""
        }
      `;

      listEl.appendChild(card);
    }

    if (filtered.length === 0) {
      listEl.innerHTML = `<div style="padding:12px; opacity:.75;">(필터 결과 없음)</div>`;
    }
  }

  applyFilter();

  statusEl.onchange = applyFilter;
  containsEl.oninput = applyFilter;

  refreshBtn.onclick = () => loadConfirmed().catch(renderError);

  listEl.onclick = async (ev) => {
    const t = ev.target;
    if (!(t && t.dataset && t.dataset.set && t.dataset.key)) return;

    const key = t.dataset.key;
    const status = t.dataset.set;

    const input = listEl.querySelector(`input[data-notes="${CSS.escape(key)}"]`);
    const notes = (input?.value || "").trim();

    t.disabled = true;
    try {
      await setStatus(key, status, notes);
      await loadConfirmed();
    } catch (e) {
      renderError(e);
    } finally {
      t.disabled = false;
    }
  };
}

async function loadConfirmed() {
  const data = await getJSON(`/ui/oov_candidates`);
  renderConfirmed(data.records || []);
}

// ----------------------
// Rejected candidates view (기존 유지 + approve 호출만 새 API로 교체)
// ----------------------

function renderRejected(items) {
  if (!items || items.length === 0) {
    out.innerHTML = `<div style="padding:12px;">(items 없음)</div>`;
    return;
  }

  out.innerHTML = `
    <div style="display:flex; gap:8px; align-items:center; padding:8px 0;">
      <div style="font-weight:700;">Rejected candidates</div>
      <div style="opacity:.7; font-size:12px;">(각 항목에서 key를 입력하고 Approve/Reject 결정 기록)</div>
    </div>
    <div id="cards" style="display:flex; flex-direction:column; gap:10px;"></div>
  `;

  const cards = document.getElementById("cards");

  for (const it of items) {
    const id = it.id || "";
    const txt = it.text || "";
    const reason = it.reason || "";
    const decided = it.decision?.status ? `✅ ${it.decision.status}` : "";

    const card = document.createElement("div");
    card.style.border = "1px solid #e5e7eb";
    card.style.borderRadius = "10px";
    card.style.padding = "10px";

    card.innerHTML = `
      <div style="display:flex; justify-content:space-between; gap:10px;">
        <div style="font-weight:700;">${escapeHtml(id.slice(0, 10))}…</div>
        <div style="font-size:12px; opacity:.75;">${escapeHtml(reason)} ${escapeHtml(decided)}</div>
      </div>

      <div style="margin-top:8px; white-space:pre-wrap;">${escapeHtml(txt)}</div>

      <details style="margin-top:8px;">
        <summary style="cursor:pointer; opacity:.8;">debug</summary>
        <pre style="white-space:pre-wrap; font-size:12px; opacity:.85;">${escapeHtml(
          JSON.stringify(
            {
              model_spans: it.model_spans || [],
              lexicon_spans: it.lexicon_spans || [],
              candidate_oov_before_filter: it.candidate_oov_before_filter,
              candidate_oov_after_filter: it.candidate_oov_after_filter,
            },
            null,
            2
          )
        )}</pre>
      </details>

      <div style="display:flex; gap:8px; margin-top:10px; align-items:center; flex-wrap:wrap;">
        <input data-key-input="${escapeHtml(id)}"
          placeholder="승인할 key (예: 멘붕, 현타)"
          style="padding:6px 8px; border:1px solid #d1d5db; border-radius:8px; min-width:220px;"
        />

        <input data-notes-input="${escapeHtml(id)}"
          placeholder="notes (선택)"
          style="padding:6px 8px; border:1px solid #d1d5db; border-radius:8px; min-width:280px;"
        />

        <button data-approve="${escapeHtml(id)}"
          style="padding:6px 10px; border-radius:8px; border:1px solid #16a34a; background:#16a34a; color:white; cursor:pointer;">
          Approve
        </button>

        <button data-reject="${escapeHtml(id)}"
          style="padding:6px 10px; border-radius:8px; border:1px solid #ef4444; background:#ef4444; color:white; cursor:pointer;">
          Reject
        </button>
      </div>
    `;

    cards.appendChild(card);
  }

  cards.onclick = async (ev) => {
    const t = ev.target;

    if (t && t.dataset && t.dataset.approve) {
      const rid = t.dataset.approve;
      const input = cards.querySelector(`input[data-key-input="${CSS.escape(rid)}"]`);
      const key = (input?.value || "").trim();
      if (!key) {
        alert("승인할 key를 입력해줘 (예: 멘붕, 현타)");
        return;
      }
      const notesInput = cards.querySelector(`input[data-notes-input="${CSS.escape(rid)}"]`);
      const notes = (notesInput?.value || "").trim();

      t.disabled = true;
      try {
        await postJSON(`/ui/oov_candidates/approve`, { key, notes });
        await postJSON(`/ui/rejected_candidates/decide`, {
          id: rid,
          status: "approved",
          note: `approved:${key}${notes ? ` # ${notes}` : ""}`,
        });
        await loadRejected();
      } catch (e) {
        renderError(e);
      } finally {
        t.disabled = false;
      }
      return;
    }

    if (t && t.dataset && t.dataset.reject) {
      const rid = t.dataset.reject;
      t.disabled = true;
      try {
        await postJSON(`/ui/rejected_candidates/decide`, {
          id: rid,
          status: "rejected",
          note: "rejected in UI",
        });
        await loadRejected();
      } catch (e) {
        renderError(e);
      } finally {
        t.disabled = false;
      }
      return;
    }
  };
}

async function loadRejected() {
  const limit = Number(limitEl?.value || 200);
  const includeProcessed = includeProcessedEl ? !!includeProcessedEl.checked : false;
  const url =
    `/ui/rejected_candidates?limit=${limit}` +
    (includeProcessed ? `&include_processed=true` : ``);

  const data = await getJSON(url);
  renderRejected(data.items || []);
}

// ----------------------
// ✅ Quick Test run
// ----------------------

async function runQuickTest() {
  if (!quickTextEl || !quickOutEl) return;
  const text = (quickTextEl.value || "").trim();
  const threshold = Number(quickThresholdEl?.value || 0.0);
  if (!text) {
    quickOutEl.textContent = "텍스트를 입력해줘.";
    return;
  }
  quickOutEl.textContent = "running...";
  const data = await postJSON(`/ui/oov_run`, { text, threshold });
  quickOutEl.textContent = JSON.stringify(data, null, 2);

  await loadConfirmed();
}

// ----------------------
// Wire up buttons
// ----------------------

document.getElementById("btnRejected").onclick = () => loadRejected().catch(renderError);
document.getElementById("btnConfirmed").onclick = () => loadConfirmed().catch(renderError);

if (includeProcessedEl) includeProcessedEl.onchange = () => loadRejected().catch(renderError);
if (btnRunEl) btnRunEl.onclick = () => runQuickTest().catch((e) => (quickOutEl.textContent = String(e)));

// default view
loadRejected().catch(renderError);

document.getElementById("btnMerge").onclick = async () => {
  try {
    const res = await postJSON("/ui/oov_merge", {
      min_count: 3,
      min_examples: 1
    });
    alert(JSON.stringify(res, null, 2));
  } catch (e) {
    alert(String(e));
  }
};
