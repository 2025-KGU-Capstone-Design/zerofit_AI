## 서버 구조
```
zerofit_AI/
├── app/
│   ├── data/                  # 모델 파일 및 전처리 데이터
│   ├── endpoints/
│   │   └── recommend.py       # 추천 API 엔드포인트
│   ├── models/
│   │   └── model.py           # 모델 로딩 함수
│   ├── services/
│   │   └── inference.py       # 추천 로직 구현
│   ├── setting/
│   │   ├── config.py          # 경로 설정 파일
│   │   └── startup.py         # 실행 시 초기화 스크립트
│   └── main.py                # FastAPI 애플리케이션 엔트리포인트
├── requirements.txt
├── .gitignore
└── README.md

```

## 의존성 설치
```
pip install -r requirements.txt
```


## 서버 실행 방법
```
uvicorn app.main:app --reload
```

## API 문서 확인
Swagger UI: http://localhost:8000/docs


## Endpoint
### /recommend 

사용자 입력과 후보 데이터셋을 기반으로 AI 모델이 예측한 결과를 바탕으로 여러 기준(종합 점수, 투자비회수기간, 투자비, 온실가스감축량)으로 상위 4개씩의의 추천 결과를 반환.

---

#### 요청 (Request)

- **HTTP Method:** POST  
- **URL:** `/recommend`  
- **Request Body (JSON):**

  ```json
  {
    "industry": '',                   // 산업군 (string)
    "targetFacilities": [],           // 대상설비 목록 (string[])
    "availableInvestment": 0,    // 투자가능금액 (백 만원, number)
    "currentEmission": 0,             // 현재 배출량 (tCO2eq, number)
    "targetEmission": 0,              // 목표 배출량 (tCO2eq, number)
    "targetRoiPeriod": 0,     // 목표 ROI 기간 (년, number)
  }
  
 #### 예시 요청 (Example Request)
 ```
  {
    "industry": "전기전자",
    "targetFacilities": [
      "공기 압축기 설비",
      "열사용설비"
    ],
    "availableInvestment": 6.6,
    "currentEmission": 100,
    "targetEmission": 90,
    "targetRoiPeriod": 0.8
  }
 ```

 #### 예시 응답 (Example Response)
 ```
  {
  {
  "solution": [
    {
      "id": null,
      "type": "total_optimization",
      "rank": 1,
      "score": 0.827,
      "industry": "전기전자",
      "improvementType": "기타설비보완",
      "facility": "열사용설비",
      "activity": "최종건조기 효율개선",
      "emissionReduction": 2.2,
      "costSaving": 1.7,
      "roiPeriod": 3.3,
      "investmentCost": 5.6,
      "bookmark": null
    },
    {
      "id": null,
      "type": "total_optimization",
      "rank": 2,
      "score": 0.826,
      "industry": "전기전자",
      "improvementType": "보온단열강화",
      "facility": "열사용설비",
      "activity": "원료 건조기 및 사출기 가열기 단열",
      "emissionReduction": 3.3,
      "costSaving": 3.1,
      "roiPeriod": 3.2,
      "investmentCost": 10,
      "bookmark": null
    },
    {
      "id": null,
      "type": "total_optimization",
      "rank": 3,
      "score": 0.824,
      "industry": "전기전자",
      "improvementType": "보온단열강화",
      "facility": "열사용설비",
      "activity": "노이용률 향상과 보온 강화로 방열손실 방지",
      "emissionReduction": 5.2,
      "costSaving": 4.1,
      "roiPeriod": 3.2,
      "investmentCost": 13.5,
      "bookmark": null
    },
    {
      "id": null,
      "type": "total_optimization",
      "rank": 4,
      "score": 0.82,
      "industry": "전기전자",
      "improvementType": "누증.누설방지",
      "facility": "보일러",
      "activity": "스팀밸브 누설증기 방지",
      "emissionReduction": 0.6,
      "costSaving": 0.8,
      "roiPeriod": 3.7,
      "investmentCost": 3,
      "bookmark": null
    },
    {
      "id": null,
      "type": "emission_reduction",
      "rank": 1,
      "score": null,
      "industry": "전기전자",
      "improvementType": "배공기열회수",
      "facility": "공조기 설비",
      "activity": "배기공기열회수 위한 열관 설치",
      "emissionReduction": 53.5,
      "costSaving": 2,
      "roiPeriod": 14.5,
      "investmentCost": 29.7,
      "bookmark": null
    },
    {
      "id": null,
      "type": "emission_reduction",
      "rank": 2,
      "score": null,
      "industry": "전기전자",
      "improvementType": "회전수제어설비도입",
      "facility": "공기 압축기 설비",
      "activity": "공기압축기 인버터 도입",
      "emissionReduction": 28.8,
      "costSaving": 24.2,
      "roiPeriod": 7.7,
      "investmentCost": 187,
      "bookmark": null
    },
    {
      "id": null,
      "type": "emission_reduction",
      "rank": 3,
      "score": null,
      "industry": "전기전자",
      "improvementType": "신설비도입",
      "facility": "공기 압축기 설비",
      "activity": "인버터적용",
      "emissionReduction": 22.9,
      "costSaving": 17.3,
      "roiPeriod": 5.8,
      "investmentCost": 100,
      "bookmark": null
    },
    {
      "id": null,
      "type": "emission_reduction",
      "rank": 4,
      "score": null,
      "industry": "전기전자",
      "improvementType": "신설비도입",
      "facility": "공기 압축기 설비",
      "activity": "2단 공기압축기로 교체",
      "emissionReduction": 18.6,
      "costSaving": 17,
      "roiPeriod": 5.9,
      "investmentCost": 100,
      "bookmark": null
    },
    {
      "id": null,
      "type": "cost_saving",
      "rank": 1,
      "score": null,
      "industry": "전기전자",
      "improvementType": "회전수제어설비도입",
      "facility": "공기 압축기 설비",
      "activity": "공기압축기 인버터 도입",
      "emissionReduction": 28.8,
      "costSaving": 24.2,
      "roiPeriod": 7.7,
      "investmentCost": 187,
      "bookmark": null
    },
    {
      "id": null,
      "type": "cost_saving",
      "rank": 2,
      "score": null,
      "industry": "전기전자",
      "improvementType": "노후설비대체",
      "facility": "보일러",
      "activity": "관리동 보일러 교체",
      "emissionReduction": 14.8,
      "costSaving": 17.5,
      "roiPeriod": 7.9,
      "investmentCost": 137.8,
      "bookmark": null
    },
    {
      "id": null,
      "type": "cost_saving",
      "rank": 3,
      "score": null,
      "industry": "전기전자",
      "improvementType": "신설비도입",
      "facility": "공기 압축기 설비",
      "activity": "인버터적용",
      "emissionReduction": 22.9,
      "costSaving": 17.3,
      "roiPeriod": 5.8,
      "investmentCost": 100,
      "bookmark": null
    },
    {
      "id": null,
      "type": "cost_saving",
      "rank": 4,
      "score": null,
      "industry": "전기전자",
      "improvementType": "노후설비대체",
      "facility": "공기 압축기 설비",
      "activity": "노후 공기압축기 교체로 압축 효율 향상",
      "emissionReduction": 16,
      "costSaving": 17,
      "roiPeriod": 6.5,
      "investmentCost": 110,
      "bookmark": null
    },
    {
      "id": null,
      "type": "roi",
      "rank": 1,
      "score": null,
      "industry": "전기전자",
      "improvementType": "보온단열강화",
      "facility": "열사용설비",
      "activity": "원료 건조기 및 사출기 가열기 단열",
      "emissionReduction": 3.3,
      "costSaving": 3.1,
      "roiPeriod": 3.2,
      "investmentCost": 10,
      "bookmark": null
    },
    {
      "id": null,
      "type": "roi",
      "rank": 2,
      "score": null,
      "industry": "전기전자",
      "improvementType": "기타설비보완",
      "facility": "열사용설비",
      "activity": "최종건조기 효율개선",
      "emissionReduction": 2.2,
      "costSaving": 1.7,
      "roiPeriod": 3.3,
      "investmentCost": 5.6,
      "bookmark": null
    },
    {
      "id": null,
      "type": "roi",
      "rank": 3,
      "score": null,
      "industry": "전기전자",
      "improvementType": "보온단열강화",
      "facility": "열사용설비",
      "activity": "노이용률 향상과 보온 강화로 방열손실 방지",
      "emissionReduction": 5.2,
      "costSaving": 4.1,
      "roiPeriod": 3.2,
      "investmentCost": 13.5,
      "bookmark": null
    },
    {
      "id": null,
      "type": "roi",
      "rank": 4,
      "score": null,
      "industry": "전기전자",
      "improvementType": "기타설비대체",
      "facility": "열사용설비",
      "activity": " 전기히터 난방기를 히트펌프형으로 대체",
      "emissionReduction": 9.7,
      "costSaving": 8.8,
      "roiPeriod": 3.4,
      "investmentCost": 30,
      "bookmark": null
    }
  ]
}

```
 