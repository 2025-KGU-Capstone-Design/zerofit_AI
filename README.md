## 서버 구조
```
project_name/
├── app/
│   ├── main.py                    
│   ├── endpoint/
│   │   └── recommend.py        # 추천 API 엔드포인트 (/recommend)
│   ├── models/
│   │   └── model.py                # 모델 인스턴스 생성 및 load_model() 함수
│   │   └── model_arch.py         # 모델 아키텍처
│   │   └── two_tower_improvement_model.pth  # 모델 파일
│   ├── setting/                       
│   │   ├── config.py		# 경로 설정
│   │   └── startup.py		# 실행 시 모델 설정
│   └── services/
│       └── inference.py            # recommend_improvements() 포함, 전처리/후처리 등 로직
├── .gitignore				
└── requirements.txt			# 의존성 설치

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
  
 #### 예시 응답 (Example Response)
 ```
  {
    "solution": [
      {
        "id": 1,
        "type": "total_optimization",
        "rank": 1,
        "score": 95.5,
        "industry": "제조업"
        "improvementType": "설비 개선",
        "facility": "보일러",
        "activity": "고효율 보일러 교체",
        "industry": "철강"
        "emissionReduction": 120.5,
        "costSaving": 3000.0,
        "roiPeriod": 2.5,
        "investmentCost": 750.0,
        "bookmark": false
      },
      {
        ...
      },

      {
        "id": 5,
        "type": "emission_reduction",
        "rank": 1,
        "score": null,
        "industry": "제조업"
        "improvementType": "설비 개선",
        "facility": "보일러",
        "activity": "고효율 보일러 교체",
        "industry": "식품"
        "emissionReduction": 130.0,
        "costSaving": 3100.0,
        "roiPeriod": 2.0,
        "investmentCost": 700.0,
        "bookmark": false
      },
      {
        ...
      },

      {
        "id": 9,
        "type": "cost_saving",
        "rank": 1,
        "score": null,
        "industry": "제조업"
        "improvementType": "설비 개선",
        "facility": "보일러",
        "activity": "고효율 보일러 교체",
        "industry": "식품"
        "emissionReduction": 125.0,
        "costSaving": 3200.0,
        "roiPeriod": 2.3,
        "investmentCost": 720.0,
        "bookmark": false
      },
      {
        "id": 10,
        "type": "cost_saving",
        "rank": 2,
        "score": null,
        "industry": "제조업"
        "improvementType": "공정 개선",
        "facility": "냉동기",
        "activity": "효율적 냉각 시스템 도입",
        "industry": "요업"
        "emissionReduction": 105.0,
        "costSaving": 2700.0,
        "roiPeriod": 2.9,
        "investmentCost": 610.0,
        "bookmark": false
      },
      {
        ...
      },

      {
        "id": 13,
        "type": "roi",
        "rank": 1,
        "score": null,
        "industry": "제조업"
        "improvementType": "설비 개선",
        "facility": "보일러",
        "activity": "고효율 보일러 교체",
        "industry": "폐기물"
        "emissionReduction": 115.0,
        "costSaving": 2900.0,
        "roiPeriod": 2.4,
        "investmentCost": 710.0,
        "bookmark": false
      },
      {
        ...
      }
    ]
  }

```
 