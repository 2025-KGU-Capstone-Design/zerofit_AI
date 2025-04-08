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

사용자 입력과 후보 데이터셋을 기반으로 AI 모델이 예측한 결과를 바탕으로 여러 기준(투자비회수기간, 투자비, 온실가스감축량)으로 상위 10개의 추천 결과를 반환.

---

#### 요청 (Request)

- **HTTP Method:** POST  
- **URL:** `/recommend`  
- **Request Body (JSON):**

  ```json
  {
      "업종": "제조업 - 식품",
      "대상설비": ["공기 압축기 설비"],
      "투자비": 50,
      "투자비회수기간": 3,
      "온실가스감축량": 18
  }
  
 #### 예시 응답 (Example Response)
 ```
 {
  "투자비회수기간_상위10": [
    {
      "대상설비": "공기 압축기 설비",
      "투자비": 50,
      "온실가스감축량": 18,
      "예측투자비": 45.2,
      "예측투자비회수기간": 2.7,
      "예측절감액": 100.0,
      "예측온실가스감축량": 20.5,
      "개선구분": "A",
      "개선활동명": "Activity A"
    }
    // ... (최대 10개 항목)
  ],
  "투자비낮은_상위10": [
    // 추천 결과 항목 (위와 유사한 구조)
  ],
  "온실가스감축량높은_상위10": [
    // 추천 결과 항목 (위와 유사한 구조)
  ]
}

```
 