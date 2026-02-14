<purpose>
김우연 c3d 파일을 v3d와 동일한 방법으로 COM, BOS, xCOM, MOS 구하기 
</purpose>

<rule>
- “(1) XCoM 계산(Hof) + (2) foot landmark 기반 BoS 폴리곤 + (3) bound까지 거리로 MoS”의 V3D 튜토리얼 로직을 그대로 적용. 
- c3d file name rule: `{date}_{name_initial}_perturb_{velocity}_{trial}.c3d`
- `종합) 주제 2 -  동일 조건 섭동 시 균형 회복 전략(step vs. non-step) 차이에 따른 수행력(performance) 변인 비교 분석.md` 파일은 내 실험한 논문 초고이다. 데이터 분석 시 참고해라. platform onset에 이후의 변화를 보고자 한다. 
- forceplate는 1개이기 때문에 작업에 있어서 사용 불가. 
- <marker_info>를 통해서 medial marker를 분석에 사용해라. 
- COM 추출 후, <v3d_com>251112_KUO_perturb_60_001_COM.xlsx</v3d_com>의 결과랑 대조해라. 상관관계가 1에 가까워야 한다. 
</rule>

<event>
platform onset, step onset의 경우 <event_file>perturb_inform.xlsm</event_file>의 'platform' sheet 참조. 
</event>

<file>

<c3d_file_info>
- units: meter
- <marker_info>: motive optitrack의 conventional full body skeleton(39)
- c3d 파일 range: <event_file> 'platform' sheet의 ["platform_onset" column - 100, "platform_offset+100]
- c3d는 motive optitrack의 conventional full body skeleton 모델사용해서 marker data를 촬영한 후, labeling, filtering 작업까지 완료. 

<좌표>
X Axis: Negative X
X축 +/- = A/P
Y Axis: Positive Z
Y축 +/- = R/L
Z Axis: Positive Y
Z축 +/- = UP/Down
</좌표>

<marker_info>
optitrack_marker_context.xml
</marker_info>

</c3d_file_info>
</file>

======

<김우연_신체계측>
다리길이	86
어깨둘레_왼	39
어깨둘레_오른	39
팔꿈치넓이_왼	8
팔꿈치넓이_오른	8
손목넓이_왼	5.5
손목넓이_오른	5.5
무릎넓이_왼	9
무릎넓이_오른	9
발목넓이_왼	6.5
발목넓이_오른	6.5
발넓이_왼	8.5
발넓이_오른	8.5
발길이_왼	240
발길이_오른	240
</김우연_신체계측>지금 첨부한 c3d는 motive optitrack의 conventional full body skeleton 모델사용해서 marker data를 촬영한 후, filtering까지 거친 motion 파일이다. 
이거가지고 COM, BOS, xCOM, MOS, 발목, 무릎 각도 구해봐. 


<other_info>
mos는 steponset 기점으로 나누지 마.forceplate도 1개이고, 시각적으로 marker를 보면서 toe off까지 검출하긴 어렵다. 어차피 분석은 step onset 직전까지만 실시하니깐 그 이후 분석에 대해 신경쓰지 말고 mos 측정 기준 변화시키지 마라. 
</other_info>