# 평창군 학교 통폐합 제언 앱  
_Pyeongchang School Integration Helper_

평창군 내 학교별 **학생 수와 위치 정보**를 기반으로,  
선택한 연도의 **학교 통폐합 가능 여부를 분석·제안**하는 Python 애플리케이션입니다.

학생 수만 보지 않고, 통폐합 시 **통학거리**까지 함께 고려하여  
실제 정책 판단에 더 가까운 결과를 보여주는 것이 목표입니다.

---

## 프로젝트 개요

- **지역**: 강원도 평창군
- **데이터**: 2022–2024년 평창군 내 각 학교별 학생 수 및 좌표 정보  
- **목표**:  
  - 학생 수 감소 지역을 고려한 통폐합 후보 학교 도출  
  - 통합 시 통학거리 증가 문제를 함께 검토  
  - “통폐합 가능 / 불가” 뿐 아니라, **불가 사유까지 명시**하는 도우미 도구

통폐합 여부는 다음 두 가지 기준을 함께 고려하여 판단합니다.

1. **최소 학생 수 기준**  
2. **통폐합 시 통학거리 기준**  
   - 위도/경도 좌표를 사용해 **유클리드 거리(Euclidean distance)** 로 계산

---

## 주요 기능

- **엑셀 기반 입력**
  - 평창군 학교 데이터가 담긴 엑셀 파일을 선택하여 분석
  - 학교명, 위도, 경도, 학생 수 등 컬럼명을 직접 지정 가능

- **유연한 통폐합 기준 설정**
  - 분석할 연도(예: `2024년도 학생수`) 선택
  - 최소 학생 수 기준 설정
  - 통폐합 시 허용 가능한 최대 통학거리 설정

- **통폐합 제언 결과 출력**
  - 통폐합 대상 후보 학교 목록
  - **통폐합 불가 학교 및 그 이유** 출력  
    (예: 통학거리 기준 초과, 학생 수 기준 미충족 등)

- **GUI 기반 사용성**
  - 시작 화면에서 파일 선택 및 기준 설정 후 “분석 실행” 버튼 클릭으로 사용
  - 결과창에서 각 학교의 통폐합 가능 여부, 불가 사유를 한눈에 확인

---

## 🛠 기술 스택
Language: Python

Data Handling: pandas (엑셀/데이터 처리)

GUI: tkinter 기반 파일 선택 및 파라미터 입력 UI

수학/거리 계산: 기본 수학 연산(위도·경도 기반 유클리드 거리)

실제 사용 라이브러리는 school_app1.py를 참고해 주세요.

## 실행 방법
프로젝트 클론

코드 복사
git clone https://github.com/425range/Pyeongchang-school-intergration-helper.git
cd Pyeongchang-school-intergration-helper
필요 패키지 설치

코드 복사
pip install -r requirements.txt
requirements.txt가 없다면, 코드에서 사용하는 라이브러리를 보고
예: pandas, openpyxl, tkinter(기본 포함) 등을 수동 설치하면 됩니다.

# 프로그램 실행 1

python school_app1.py
실행 후 나타나는 GUI에서:
<img width="960" height="586" alt="image" src="https://github.com/user-attachments/assets/78450ea4-7d8d-413d-b813-31f2da51116c" />
- 분석할 엑셀 파일 선택

- 학교명, 위도, 경도, 학생 수 등 컬럼명 입력

- 분석 연도에 해당하는 학생 수 열 선택

예: 2024년도 학생수

최소 학생 수, 최대 통학거리 기준 설정

- 분석 실행 버튼 클릭
<img width="960" height="580" alt="image" src="https://github.com/user-attachments/assets/d93813be-05e2-432b-9cb3-50ac7066ed9b" />

## 입력 데이터 형식 (예시)
엑셀 파일에는 대략 다음과 같은 열이 포함됩니다.

학교명 / 위도 /경도	/ 2022년도 학생수 /	2023년도 학생수 /	2024년도 학생수

○○초등학교	/ 37.123456	/ 128.123456	/ 32	/ 28	/ 21

△△중학교 /	37.234567 /	128.234567	/ 85	/ 79 /	71

사용 시, 시작 화면에서 각 열의 이름을 정확히 지정해야
올바르게 분석이 진행됩니다.

## 알고리즘 개요
데이터 로딩 : 선택한 엑셀 파일을 불러와 DataFrame 형태로 저장

필터링 : 선택한 연도의 학생 수 열 기준으로 최소 학생 수 미만 학교 탐색

거리 계산 : 각 학교 간 위도·경도 정보를 기반으로 유클리드 거리 계산

통폐합 가능성 판단 : 통합 시 통학거리 기준을 초과하는지 여부 평가, 기준을 만족하지 못하는 학교에 대해 통폐합 불가 사유 함께 출력

결과 출력 : GUI 결과창에 학교별 통폐합 가능 여부와 사유를 표시

# 프로그램 실행 2 (ARIMA 분석)

<img width="500" height="300" alt="image" src="https://github.com/user-attachments/assets/327aa578-3366-4efc-945c-96823dd8d15c" />  

PACF/ACF 분석과 Auto-ARIMA function을 통해 최적의 파라미터를 탐색하고,  
평창, 양양, 홍천, 횡성군 중 가장 청년 인구 (8~19세)의 감소세가 큰 도시를 확인합니다.  

## 입력 데이터 형식 (예시)
<img width="500" height="250" alt="image" src="https://github.com/user-attachments/assets/acafaffa-3a17-455e-acd5-af34d3028a6f" />  

행 : 지역이름  
열 : 연도  
  
## 알고리즘 개요

PACF/ACF 함수를 통해 주어진 시계열 인구 데이터를 예측하고 그래프로 시각화하여 적합한 모델을 추천합니다.  
이후 auto-arima 를 통해 얻은 최적의 파라미터를 ARIMA 예측 모델에 입력하여 각 지역별 인구 예측 결과를 출력합니다.  


## 향후 개선 방향
ARIMA 등 시계열 모델을 활용한 미래 학생 수 예측 향후 몇 년을 내다본 중장기 통폐합 시나리오 분석

평창군 외 다른 지자체 데이터로 일반화 가능하도록 구조 개선 결과를 지도 기반 시각화(예: Folium, 지도로 학교 위치/통학거리 표시)

-----
# Pyeongchang School Integration Helper
This program analyzes student population and location data for schools in Pyeongchang County and outputs recommendations on possible school consolidations for a selected year.

Consolidation decisions are based on student enrollment numbers and the commuting distance required if schools are merged. 
Distances are calculated using latitude and longitude through Euclidean distance.

The current dataset includes student counts for schools in Pyeongchang County from 2022 to 2024.

## How the Program Works
<img width="960" height="586" alt="image" src="https://github.com/user-attachments/assets/78450ea4-7d8d-413d-b813-31f2da51116c" />

### 1. Start Screen

The start screen allows users to:

Select an Excel file containing the school data

Specify the names of each relevant column

Choose the student count column for the year to be analyzed (e.g., “2024 Student Count”)

Set consolidation criteria such as:

Minimum number of students

Maximum allowed commuting distance after consolidation

These settings help evaluate potential issues and guide suggestions during the analysis.

### 2. Input Phase
<img width="960" height="580" alt="image" src="https://github.com/user-attachments/assets/d93813be-05e2-432b-9cb3-50ac7066ed9b" />

After configuring the consolidation criteria and specifying the column names, clicking Run Analysis performs the data evaluation.

The program identifies schools that do not meet the required standards and displays them in the results window.

From the output, users can observe:

Which schools are candidates for consolidation

Which schools cannot be consolidated and the reasons why

Detailed information for each school based on the configured criteria

# Conclusion

Through this project, I gained hands-on experience with data analysis techniques and learned how to design a rule-based classification model using custom-defined policies.

In the future, the system can be upgraded using time-series forecasting models such as ARIMA, allowing predictions of future student population trends and enabling more advanced school consolidation recommendations.
