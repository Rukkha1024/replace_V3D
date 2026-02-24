모두 **Hof (2005, 2008)의 기본 XCoM 개념**을 기반으로 하며, subject-specific foot length나 BOS 길이로 나누어 키·다리 길이 차이를 보정합니다. 주로 standing/treadmill perturbation 연구에서 initial posture, early response stability, stepping threshold 분석에 쓰입니다.

아래는 **주요 논문별 list**와 **정확한 수식 + 정의**입니다 (Methods 섹션에서 직접 추출/인용 기반).

### 1. Van Wouwe et al. (2021) — 원본 논문 (standing support-surface backward translation)
**목적**: trial-by-trial intrasubject variability 설명, early response (0–300 ms) stability 측정.  
**BOS 정의**: anterior functional BOS = static trial에서 ankle joint (talus origin)에서 toe segment origin까지의 horizontal distance (scaled OpenSim model).  
**수식**:
\[
xCOM = COM_x + \frac{\dot{COM}_x}{\sqrt{g / l}}
\]
\[
\frac{xCOM}{BOS} = \frac{xCOM}{BOS_{length}}
\]
- \(COM_x, \dot{COM}_x\): whole-body COM horizontal position & velocity (ankle joint 기준, platform motion 보정).  
- \(l\): static trial에서의 whole-body COM vertical height.  
- \(g\): gravity (9.81 m/s²).  
- 사용 시점: perturbation onset (xCOM/BOS_onset) & 300 ms (xCOM/BOS_300ms).  
- **특징**: 완전 dimensionless (0~1 범위), subject 간 비교에 최적화.

### 2. Salot et al. (2016) — "Reactive Balance in Individuals With Chronic Stroke..." (Phys Ther)
**목적**: stroke 환자 vs. healthy control의 backward falling 시 postural stability (step lift-off / touchdown 때).  
**BOS 정의**: foot length (heel-to-toe) 또는 anterior-posterior BOS boundary.  
**수식** (Bhatt 연구그룹 표준):
\[
XCOM = x_{COM} + \frac{\dot{x}_{COM}}{\sqrt{g / l}}
\]
\[
\frac{XCOM}{BOS} = \frac{XCOM - BOS_{rear\ or\ boundary}}{foot\ length}
\]
- \(l\): leg length 또는 COM height (subject-specific).  
- 추가로 velocity 항: \(\dot{XCOM}/BOS\)도 함께 사용.  
- **특징**: step TD (touchdown) 시점에서 가장 불안정한 순간 측정. Stroke 그룹에서 posterior XCOM/BOS 값이 더 크다고 보고.

### 3. Joshi et al. (2018) — "Reactive balance to unanticipated trip-like perturbations..." (treadmill-based, aging & stroke)
**목적**: trip-like perturbation에서 fall risk 메커니즘 분석 (lift-off & touchdown 시점).  
**BOS 정의**: individuals' foot length (heel-to-toe distance).  
**수식** (snippet에서 직접 인용):
\[
XCOM/BOS = \frac{(absolute\ COM\ position - BOS\ position)}{foot\ length}
\]
\[
VCOM/BOS = \frac{(COM\ velocity - BOS\ velocity)}{\sqrt{g \times body\ height}}
\]
- XCOM 자체는 Hof 공식 사용.  
- **특징**: position은 foot length로 normalize → dimensionless. Velocity는 √(g h)로 normalize (h = body height). Young control, older control, stroke 그룹 비교에 사용.

### 4. Patel et al. (2015) & Bhatt 그룹 후속 연구 (e.g., Adaptation to large-magnitude treadmill-based perturbations)
**목적**: repeated slip/trip perturbation adaptation, compensatory step length와 연계.  
**BOS 정의**: rear of BOS (heel 위치) 기준 relative position.  
**수식**:
\[
XCOM/BOS = \frac{absolute\ AP\ COM\ position\ relative\ to\ rear\ of\ BOS}{foot\ length}
\]
- 기본 XCOM = \(x + \dot{x} / \sqrt{g/l}\) (l = COM height).  
- **특징**: walking perturbation에서 자주 사용. Rear BOS 기준으로 계산 → backward loss-of-balance 예측에 강함. Adaptation trial에서 XCOM/BOS가 감소한다고 보고.

### 5. 기타 Bhatt/Pai 연구그룹 공통 패턴 (2006~2020대 다수 논문, e.g., slip exposure adaptation)
**수식** (가장 일반적 형태):
\[
XCOM/BOS = \frac{XCOM_{AP} - BOS_{rear}}{foot\ length\ (or\ BOS_{AP\ length})}
\]
- Velocity 버전도 동일하게 normalize: \(\dot{XCOM}/\sqrt{g \cdot h}\).  
- **특징**: foot length normalization이 표준 → inter-subject variability 최소화. MoS (Margin of Stability = XCOM – BOS boundary)와 함께 쓰이는 경우 많음.

**공통 특징 요약**  
- **기본 XCOM**: 항상 Hof 공식 (position + velocity term).  
- **Normalization 이유**: raw xCOM (cm 단위)은 키 차이 때문에 비교 불가 → foot length 또는 subject-specific BOS length로 나누어 dimensionless 값으로 만듦.  
- **사용 시점**: 주로 perturbation onset, early response end (∼300 ms), step lift-off/touchdown.  
- **장점**: Van Wouwe처럼 correlation 분석 (initial posture vs. maximal trunk lean 등)에 최적.

이 방법들은 거의 모두 OpenSim이나 Vicon + force plate 데이터로 계산되며, l (pendulum length)는 static trial에서 결정합니다. 더 특정 논문 (e.g., 2020년 이후 최신)이나 전체 PDF Methods 전체 인용, 또는 코드 예시가 필요하시면 말씀해주세요!