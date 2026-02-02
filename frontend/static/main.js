// frontend/static/main.js

let emotionChart = null; // Chart.js 인스턴스

document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("analyze-form");
  const textInput = document.getElementById("diary-text");
  const textCounter = document.getElementById("text-counter");
  const errorBox = document.getElementById("error-box");
  const loadingIndicator = document.getElementById("loading-indicator");
    const submitBtn = form?.querySelector('button[type="submit"]');
  // 글자 수 카운터
  if (textInput && textCounter) {
    textInput.addEventListener("input", () => {
      textCounter.textContent = `(${textInput.value.length}자)`;
    });
  }

  // ===== 1) 첫 페이지: /analyze 호출 후 /result로 이동 =====
  if (form) {
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      hideError(errorBox);

      const nameInput = document.getElementById("student-name");
      const ageInput = document.getElementById("student-age");
      const useLlmInput = document.getElementById("use-llm");

      const name = (nameInput?.value || "").trim();
      const ageValue = ageInput?.value || "";
      const text = (textInput?.value || "").trim();
      const useLlm =
        useLlmInput && useLlmInput.type === "checkbox"
          ? !!useLlmInput.checked
          : (useLlmInput?.value || "true") === "true";

            // 먼저 내용 비었는지 체크
      if (!text) {
        showError(errorBox, "내용을 적어주세요");
        return;
      }

      // 그 다음 이름/나이 체크
      if (!name || !ageValue) {
        showError(errorBox, "이름, 나이를 입력해 주세요.");
        return;
      }

      const age = Number(ageValue);
      if (Number.isNaN(age) || age <= 0) {
        showError(errorBox, "나이는 양의 정수로 입력해 주세요.");
        return;
      }

      // 성별
      let gender = "기타";
      const genderRadio = document.querySelector("input[name='gender']:checked");
      if (genderRadio && genderRadio.value) {
        gender = genderRadio.value;
      } else {
        const genderSelect = document.getElementById("gender");
        if (genderSelect && genderSelect.value) {
          gender = genderSelect.value;
        }
      }

      const payload = {
        name,
        age,
        gender,
        text,
        use_llm: useLlm,
      };

      showLoading(loadingIndicator, true);

      try {
        const response = await fetch("/analyze", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });

        const data = await response.json();

        if (!response.ok) {
          const message = data?.detail || "서버 오류가 발생했습니다.";
          showError(errorBox, message);
          return;
        }

        let wrapped;

        // 새 포맷: {status:"ok", result:{...}}
        if (data.status === "ok" && data.result) {
          wrapped = data;
        }
        // 옛 포맷 호환: {student:{...}, summary:...}
        else if (data.student && data.summary) {
          wrapped = {
            status: "ok",
            use_llm: useLlm,
            result: data,
          };
        } else {
          showError(errorBox, "알 수 없는 응답 형식입니다.");
          return;
        }

        // 결과를 sessionStorage에 저장하고 result 페이지로 이동
        sessionStorage.setItem("analysisResult", JSON.stringify(wrapped));
        window.location.href = "/result";
      } catch (err) {
        console.error(err);
        showError(errorBox, "요청 중 네트워크 오류가 발생했습니다.");
      } finally {
        showLoading(loadingIndicator, false);
      }
    });
  }

  // ===== 2) 결과 페이지: 저장된 결과 읽어서 UI 채우기 =====
  const resultSection = document.getElementById("result-section");
  if (resultSection) {
    const stored = sessionStorage.getItem("analysisResult");
    if (stored) {
      try {
        const data = JSON.parse(stored);
        updateResultUI(data);
      } catch (e) {
        console.error("저장된 분석 결과 파싱 오류:", e);
      }
    }
  }
});

/* ========= 결과 UI 업데이트 (result.html 전용) ========= */

function updateResultUI(data) {
  const result = data.result || {};

  // 1) 학생 정보
  const student = result.student || {};
  setText("result-student-name", student.name ?? "-");
  setText("result-student-age", student.age ?? "-");
  setText("result-student-gender", student.gender ?? "-");
  setText("result-use-llm", data.use_llm ? "예" : "아니오");

  // 2) 규칙 기반 지표
  const detail = result.detail || {};
  const textLen = detail.text_length ?? null;
  const negCount = detail.negative_word_count ?? null;

  setText("detail-text-length", textLen ?? "-");
  setText("detail-negative-count", negCount ?? "-");

  // 1000자당 부정어휘 개수
  let densityText = "-";
  if (textLen && negCount != null) {
    const density = (negCount / textLen) * 1000;
    densityText = `1000자당 ${density.toFixed(1)}개`;
  }
  setText("detail-negative-frequency", densityText);

  // 전체 부정 비율 (%)
  let ratioText = "-";
  if (typeof result.negative_ratio === "number") {
    ratioText = `${(result.negative_ratio * 100).toFixed(1)}%`;
  }
  setText("detail-negative-ratio", ratioText);

  // 3) 감정 분포 → 그래프 + 순위 리스트
  const scores = result.scores || {};
  renderEmotionChart(scores);

  const emotionList = document.getElementById("cg-emotion-list");
  if (emotionList) {
    emotionList.innerHTML = "";
    const entries = Object.entries(scores);
    const totalScore = entries.reduce((sum, [, v]) => sum + (v || 0), 0);

    entries
      .sort((a, b) => (b[1] || 0) - (a[1] || 0))
      .forEach(([emotion, score], idx) => {
        const li = document.createElement("li");
        const ratio =
          totalScore > 0 ? ((score / totalScore) * 100).toFixed(1) : "-";
        li.textContent = `${idx + 1}. ${emotion}: ${score}점 (${ratio}%)`;
        emotionList.appendChild(li);
      });
  }

  // 4) LLM summary(JSON) → 개조식 / 서술식 나누기
  const ruleSummaryEl = document.getElementById("cg-rule-summary");
  const feedbackEl = document.getElementById("cg-llm-feedback");
  const summaryRaw = result.summary;

  let summaryObj = null;

  try {
    if (typeof summaryRaw === "string") {
      let clean = summaryRaw.trim();
      // 혹시 모를 ```제거 (예방용)
      clean = clean.replace(/```json/gi, "").replace(/```/g, "").trim();
      summaryObj = JSON.parse(clean);
    } else if (typeof summaryRaw === "object" && summaryRaw !== null) {
      summaryObj = summaryRaw;
    }
  } catch (e) {
    console.error("LLM JSON parse 실패:", e, summaryRaw);
    summaryObj = null;
  }

  if (summaryObj) {
    // ─ 개조식 분석 ─
    const kj = summaryObj["개조식_분석"] || {};
    if (ruleSummaryEl) {
      ruleSummaryEl.innerHTML = "";

      const ul = document.createElement("ul");
      ul.className = "cognitive-list small";

      const addItem = (text) => {
        if (typeof text !== "string") return;
        const trimmed = text.trim();
        if (!trimmed) return;
        const li = document.createElement("li");
        li.textContent = `${trimmed}`;
        ul.appendChild(li);
      };

        // 여러 버전의 키를 모두 지원하기 위한 helper
      const pick = (obj, ...keys) => {
        for (const k of keys) {
          const v = obj[k];
          if (typeof v === "string" && v.trim()) return v.trim();
        }
        return null;
      };
      
      // 1줄: 부정어휘/감정 밀도 쪽 요약
      addItem(
        pick(
          kj,
          "감정_밀도_요약",   // 현재 user 프롬프트
          "어휘_사용_요약",   // 이전 버전 키
          "어휘_사용_분석"    // system 프롬프트 JSON 스키마
          )
        );
      // 2줄: 내용/사건 요약
      addItem(
        pick(
          kj,
          "내용_핵심_요약",
          "감정_분포_특성"    // 혹시 이 키에 들어올 수도 있어서
        )
      );
      // 3줄: 학생 이해/글 특성 요약
      addItem(
        pick(
          kj,
          "학생_이해_시사점",
          "글_특성_요약",
          "학생_경험_요약"
        )
      );
      
      if (ul.childElementCount > 0) {
        ruleSummaryEl.appendChild(ul);
      } else {
        ruleSummaryEl.textContent = "개조식 분석 결과가 없습니다.";
      }
    }

    // ─ 서술식 피드백 ─
    const sj = summaryObj["서술식_피드백"] || {};
    if (feedbackEl) {
      feedbackEl.innerHTML = "";
      let index = 1;

      const addSection = (title, text) => {
        if (typeof text !== "string" || !text.trim()) return;

        const h = document.createElement("h4");
        h.textContent = `${index}. ${title}`;
        feedbackEl.appendChild(h);

        const p = document.createElement("p");
        p.textContent = text.trim();
        feedbackEl.appendChild(p);

        index += 1;
      };

      addSection("내용적 특성", sj["내용적_특성"]);
      addSection("심리적 특성", sj["심리적_특성"]);
      addSection("구조적 특성", sj["구조적_특성"]);

      if (!feedbackEl.hasChildNodes()) {
        feedbackEl.textContent = "서술식 피드백이 없습니다.";
      }
    }
  } else {
    // JSON 파싱 실패 시: 통째로 서술식에만 보여주기
    if (ruleSummaryEl) {
      ruleSummaryEl.textContent =
        "개조식 분석 정보가 JSON 형식이 아니라서 분리하지 못했습니다.";
    }
    if (feedbackEl) {
      feedbackEl.textContent =
        typeof summaryRaw === "string" && summaryRaw.trim()
          ? summaryRaw
          : "요약 결과가 비어 있습니다.";
    }
  }
}

/* ========= 공통 헬퍼 ========= */

function setText(id, value) {
  const el = document.getElementById(id);
  if (el) el.textContent = value;
}

function showError(box, message) {
  if (!box) return;
  box.textContent = message;
  box.classList.remove("hidden");
}

function hideError(box) {
  if (!box) return;
  box.textContent = "";
  box.classList.add("hidden");
}

function showLoading(indicator, isLoading) {
  if (!indicator) return;
  if (isLoading) indicator.classList.remove("hidden");
  else indicator.classList.add("hidden");
}

/**
 * 감정 분포 그래프 (Radar Chart: 육각형 형태)
 * @param {Object.<string, number>} scores
 */
function renderEmotionChart(scores) {
  const canvas = document.getElementById("emotion-chart");
  if (!canvas) return;

  const ctx = canvas.getContext("2d");

  if (emotionChart) {
    emotionChart.destroy();
    emotionChart = null;
  }

  const labelMap = {
    anger: "분노",
    anxiety: "불안",
    sadness: "슬픔",
    depression: "우울",
    frustration: "좌절",
    lonely: "외로움",
  };

  const labels = Object.keys(scores || {}).map(
    (key) => labelMap[key] || key
  );
  const values = Object.values(scores || {});

  if (!labels.length) return;

  emotionChart = new Chart(ctx, {
    type: "radar",
    data: {
      labels,
      datasets: [
        {
          label: "감정 점수",
          data: values,
          fill: true,
          backgroundColor: "rgba(255, 99, 132, 0.2)",
          borderColor: "rgba(255, 99, 132, 1)",
          pointBackgroundColor: "rgba(255, 99, 132, 1)",
          pointBorderColor: "#fff",
          pointHoverBackgroundColor: "#fff",
          pointHoverBorderColor: "rgba(255, 99, 132, 1)",
        },
      ],
    },
    options: {
      responsive: true,
      scales: {
        r: {
          beginAtZero: true,
          suggestedMax: 30,
        },
      },
      plugins: {
        legend: { display: false },
      },
    },
  });
}