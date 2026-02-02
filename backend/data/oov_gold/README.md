# OOV Gold 데이터 규격

각 파일(train/dev/test)은 JSON Lines(.jsonl) 형식이며, 한 줄이 한 문서를 의미합니다.

## 스키마
```json
{
  "id": "doc_0001",
  "text": "원문 전체 텍스트...",
  "oov_spans": [
    {"start": 18, "end": 26, "label": "NEG_OOV"}
  ]
}
```

- `start/end`: 원문 `text` 기준 **char index**, `end`는 exclusive 입니다.
- 오프셋 정합을 위해 `text`에 대해 임의의 trim/개행변환/정규화 등을 하지 마세요.
- `label`은 현재 최소 `NEG_OOV`만 사용합니다.
