# zerofit_AI

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