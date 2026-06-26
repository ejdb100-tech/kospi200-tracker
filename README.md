# 코스피200 50일 이격도 · 메모리 MDD 트래커

원본 [이그전 50일 이격도 트래커](https://andy-0401.github.io/kospi-ma-disparity/)를 두 가지로 바꾼 버전:

1. **이격도 기준을 코스피 → 코스피200**으로 변경
2. **SK하이닉스·삼성전자 낙폭(drawdown)·MDD 모니터** 추가

구조는 원본과 동일하게 **GitHub Pages 정적 사이트 + GitHub Actions cron**입니다. 서버 불필요.

```
index.html            정적 프론트엔드 (Chart.js, data.json을 fetch)
update_data.py        FinanceDataReader로 data.json 생성
data.json             cron이 매 거래일 갱신 (없으면 index.html 내장 샘플 표시)
.github/workflows/update.yml   KST 12:00 / 15:40 자동 실행
```

## 배포 (5분)

```bash
# 1. 새 repo 만들고 이 파일들 푸시
git init && git add . && git commit -m "init" && git push

# 2. 로컬에서 첫 data.json 한 번 생성 (선택)
pip install finance-datareader pandas
python update_data.py            # data.json 생성
git add data.json && git commit -m "first data" && git push
```

GitHub repo → **Settings → Pages → Source: main branch / root** 으로 설정하면
`https://<유저명>.github.io/<repo>/` 에서 열립니다.
Actions 탭에서 `update-data` 워크플로를 한 번 수동 실행(`Run workflow`)하면 즉시 실데이터로 채워집니다.

> `data.json`이 아직 없으면 페이지는 `index.html`에 내장된 **합성 샘플**을 보여주고 상단에 경고 배너를 띄웁니다.

## 지표 정의

- **이격도** = 코스피200 종가 ÷ 50일 이동평균 × 100
  - ≥130 과열 · 120–130 경계 · 105–120 정상 · ≤105 과열해소
- **낙폭** = 종가 ÷ (선택 기간 내) 누적 전고점 − 1
- **MDD** = 선택 기간 내 낙폭의 최저점. 전고점은 기간 시작부터의 running-peak(`cummax`) 기준이라, 3M/6M/1Y/2Y/5Y 토글에 따라 값이 달라집니다.

## 알아두면 좋은 한계 (직접 쓰실 때)

- **임계값 130/120/105는 원 이론이 광의의 코스피에서 잡은 값**입니다. 코스피200은 삼성전자·SK하이닉스 등 대형주/반도체 비중이 더 높아 이격도 분포가 다릅니다 — 같은 130이라도 의미가 같지 않을 수 있습니다.
- **이격도는 변동성 정규화가 안 된 고정 임계값**입니다. 저변동 그라인드로 도달한 130과 급등 스파이크로 도달한 130은 다릅니다. 정량 신호로 쓰려면 실현변동성으로 스케일링한 z-score 형태가 더 안전합니다.
- **순환참조 주의.** 코스피200 이격도와 SK하이닉스/삼성전자 낙폭은 정보가 상당히 겹칩니다(두 종목이 코스피200의 큰 비중). 독립 신호로 합산하지 마세요.

## 임계값/종목 바꾸기

- 종목 추가/교체: `update_data.py`의 `STOCKS` 딕셔너리와, 필요시 `index.html`의 색상(`--hynix`,`--sec`)만 수정.
- 다른 지수: `KOSPI200_CODE`를 바꾸면 됩니다 (코스피=1001, 코스피200=1028, 코스닥150=2203 등).
- 임계값: `index.html`의 `regimeOf()` 함수 한 곳에서 관리.

## 신호 검증 하네스 — `validate_disparity.py`

고정 임계값(130/120/105)이 눈대중이라는 한계를 정량적으로 메우는 도구. **매매 시스템이 아니라 검증용**입니다.

테스트하는 질문: *코스피200이 과열(이격도 높음)일 때 실제로 선행수익률이 낮아지는가? 변동성 스케일링이 신호를 개선하는가?*

비교 신호 3종 × 호라이즌 3종(5/20/60일):
- `disparity` — 원본 이격도 (고정 임계 130/120/105 버킷 표 포함)
- `z_vol` — 변동성 스케일 = (close/ma50 − 1) / (σ_d·√50). "50일 변동성 몇 배만큼 위에 있나"
- `z_roll` — 이격도의 롤링 z-score(적응형 임계값 대용)

누설 방지: **purged walk-forward**(버킷 경계는 과거 train에서만, 평가는 미래 test에서) + **embargo**(선행수익 창 h일 격리). 유의성은 중첩(overlap) 보정을 위해 **circular-shift 순열검정**과 **moving-block 부트스트랩**으로 산출하고, 신호×호라이즌 9개에 대한 **BH(FDR) 보정 p값**을 함께 보고합니다.

```bash
pip install finance-datareader pandas numpy matplotlib
python validate_disparity.py --demo            # 합성 데이터로 동작 확인(네트워크 불필요)
python validate_disparity.py --start 2014-01-01 --plot   # 코스피200 실데이터
```

출력 읽는 법:
- `spearman < 0` = 더 extended일수록 선행수익률 낮음(이론과 일치 방향).
- `p_perm_BH < 0.05` 이고 `spread_IS`·`spread_OOS` 부호가 spearman과 일치하면 '진짜 신호'.
- `z_vol`의 |spearman|·유의성이 `disparity`보다 크면 변동성 스케일링이 실제로 개선한 것.
- `spread_IS`와 `spread_OOS`가 크게 갈리면 과적합/누설 의심.
- 결과는 `validation_results.csv`, 버킷별 선행수익률 차트는 `validation_buckets.png`로 저장.

> 검증에서 `z_vol`이 이기면, `index.html`의 `regimeOf()`를 z_vol 분위 기반으로 바꿔 사이트 임계값을 적응형으로 교체할 수 있습니다.

데이터: KRX/네이버 무료 공개(FinanceDataReader). 투자 권유 아님 · 판단 책임은 본인.
