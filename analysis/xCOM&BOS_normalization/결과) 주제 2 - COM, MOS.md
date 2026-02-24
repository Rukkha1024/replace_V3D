---
cssclass: clean-embeds
date created: 2026-01-19. 00.31
---


# 1. 가설 


# 2. 결론 

- DV1 기준으로 보면, step이 nonstep보다 **더 앞**에 있는 게 아니라 오히려 **더 뒤(후방)**에 있음. 
	- platform onset: step < nonstep (`0.5091` vs `0.5715`, Estimate `-0.0440`, `***`)
	- step onset: step < nonstep (`0.2354` vs `0.3232`, Estimate `-0.0893`, `***`)
- 즉 platform onset 시점의 초기 xCOM-BOS 상대 위치 차이가 strategy(step vs nonstep)와 연관되어 있음. 


# 3. results 

## DV1 통계결과 (LMM)

- 모델: `DV1 ~ step_TF + (1|subject)`  
- DV1 정의: `(xCOM_hof - BOS_rear) / foot_length`

- platform onset
	- Step (M±SD): `0.5091±0.0596`
	- Nonstep (M±SD): `0.5715±0.0654`
	- `step_TFstep` Estimate: `-0.0440` (`***`)
- step onset
	- Step (M±SD): `0.2354±0.1720`
	- Nonstep (M±SD): `0.3232±0.0996`
	- `step_TFstep` Estimate: `-0.0893` (`***`)

- 방향 해석
	- DV1 값이 작을수록 BOS rear 대비 xCOM이 더 후방.
	- 따라서 두 이벤트 모두 step군이 nonstep군보다 상대적으로 후방 위치.


## plot으로 확인했을 때 

### COM 
![](https://i.imgur.com/7d8Qh1M.png)

- COM X
	- step_onset 이후로 극명하게 나뉨. 
	- vCOM, xCOM은 그래도 onset - step_onset 사이에 살짝 차이가 있음. nonstep 값이 좀 더 높네. 
- COM Y: 애초에 stepping의 영역이다 보니깐 별로 의미가 없는 것 같네. 
- COM Z: 둘이 차이 못 느낌 

### MOS 

![](https://i.imgur.com/4clnWYI.png)

- MOS 부호 설명
	- 양수(+): xCOM이 BOS 안에 있음. 
	- 음수(-): xCOM이 BOS 밖으로 나감. 
- MOS AP: nonstep이 더 안정적이다? <!-- 근데 이게 끝? 해석을 못하겠음  -->
- MOS ML: 얘도 모르겠음. <!-- 선행연구나 읽자 -->




# 4. key papers와의 결과 일치도 

## COM 

- key paper: [[@Interactions between initial posture and task-level goal explain experimental variability in postural responses to perturbations of standing balance|Van Wouwe et al., 2021]]
- 일치점: Van Wouwe et al. (2021)이 제시한 것처럼, 초기 자세/초기 xCOM-BOS 상태가 전략 차이와 연결된다는 해석과 맞음.
- 차이점: Van Wouwe는 onset-early window(예: 0-300 ms) 중심이고, 본 분석은 `platform onset`, `step onset` 이벤트 기반이라는 점이 다름.

##  key papers


1. [[@Effects of the type and direction of support surface perturbation on postural responses|Chen et al., 2014]]

2. [[@Interactions between initial posture and task-level goal explain experimental variability in postural responses to perturbations of standing balance|Van Wouwe et al., 2021]]
	1. 섭동 전 xCOM/BOS의 값이 strategy를 변경하는데 있어서 중요하다고 이야기함. 
	2. 유저의 연구에 따르면 step vs. nonstep 간에 차이가 있음. 
