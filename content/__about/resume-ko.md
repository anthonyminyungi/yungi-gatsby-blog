---
title: 'about'
date: 2021-7-11 23:41:30
lang: 'ko'
showToc: false
---
<div align="right"><sub><i>최근 수정한 날짜: 2021.07.11</i></sub></div>

# 민윤기 (Anthony)

## 소개

**저는 이런 개발자입니다.**

* 호기심 많고, 도전적이며 문제 해결 과정에 열정적입니다.
* 적어도 개발에 있어서만큼은 Grit을 발휘합니다.
* 작지만 큰 차이를 만든다고 믿는 디테일에 몰입합니다.
* 좋은 동료와 함께하고 싶고, 저 또한 타인에게 좋은 동료가 되고자 합니다.
* 항상 배우려는 자세로 임하고, 지식과 경험 앞에 겸손하려 합니다.

|            |                                      |
| :--------: | :----------------------------------- |
| **Github** | <https://github.com/anthonyminyungi> |
|  **Blog**  | <https://yungis.dev>                 |
| **Email**  | <yungi.anthony.min@gmail.com>        |

<br/>

# 학업

## 충남대학교 컴퓨터공학과

* (2013.03) 화학공학과 입학
* (2015.03) 컴퓨터공학과 전입
* (2021.08) 학부 졸업

<br/>

# 업무 경험

<!-- ## N-Tech Service (NTS)

|               |                |
| :-----------: | -------------- |
|   **기간**    | 2021.07 ~ 현재 |
| **역할/직책** | 인턴           |
 -->

## (주) 하얀마인드

|               |                    |
| :-----------: | ------------------ |
|   **기간**    | 2020.06 ~ 2020.08  |
| **역할/직책** | 백오피스 개발 인턴 |

### COVID-19 알림 챗봇 제작

* 2020.07 ~ 2020.08
* Facebook workplace chat API, Firebase functions, Typescript, Axios, Cheerio

#### 상세설명

* 코로나19로 인해 사내에서 규정하는 원격근무 여부 결정 및 현황 정보 전달을 용이하게 하기 위해 직접 회사에 제안 후 진행한 사이드 프로젝트.
* 보건복지부, 대전광역시 코로나19 현황판 페이지를 크롤링하여 메신저의 단체방으로 매일 오전에 전송하도록 구현.

### 문서 동시편집 방지 기능 구현

* 2020.06 ~ 2020.08
* Firebase, React, react-admin, GraphQL

#### 상세설명

* 기존 백오피스의 유저 간 동시 편집으로 인한 덮어쓰기 이슈를 방지하기 위해 인턴십 기간동안 단독 장기 프로젝트로 진행.
* Firestore에 신규 컬렉션 추가 및 그에 대응하는 서버사이드 API scheme과 resolver를 GraphQL로 구성하고 Transaction을 통해 편집 상태 변경 요청을 처리.
* 문서 상세 뷰에서 편집 상태에 따른 UI와 편집 상태 업데이트 및 해지를 Mutation 할 수 있도록 구현.

## (주) 하얀마인드

|               |                    |
| :-----------: | ------------------ |
|   **기간**    | 2019.12 ~ 2020.03  |
| **역할/직책** | 백오피스 개발 인턴 |

### 백오피스 유지보수 및 기능 개선

* 2020.01 ~ 2020.03
* Firebase, React, react-admin, Material-UI, Algolia

#### 상세설명

* 백오피스의 컨텐츠 문서 생성 페이지의 태그 선택 기능을 기존의 가독성이 떨어지는 Dropdown 방식에서 Checkbox를 활용한 UI로 변경하는 작업 수행.
* 백오피스에 검색 기능을 추가하는 프로젝트 진행.
  * Algolia를 활용해 컬렉션 별 검색 인덱스를 생성하고 Firesotre의 데이터 변경이 이루어질때마다 검색 인덱스를 업데이트 하도록 트리거 구성.
  * API call 최소화를 위해 Search Input에 대해 Debounce 적용.
* 이외 백오피스 프로젝트의 크고 작은 이슈 해결 업무 수행.

### 사내 홈페이지 리뉴얼

* 2019.12 ~ 2020.03
* React, Google map

#### 상세설명

* 기존 Jekyll로 제작된 회사 홈페이지를 새로운 React SPA로 리뉴얼한 프로젝트.
* 디자이너와 1:1 협업을 통해 초기 개발부터 배포까지 진행.
* <https://hayanmind.com>

<br/>

# 오픈소스 기여

## React.js 공식 문서 한국어 번역

* Github: <https://github.com/reactjs/ko.reactjs.org/pull/222>
* 문서: <https://ko.reactjs.org/docs/error-boundaries.html>

<br/>

# 프로젝트

## 0Auth (Zero-Auth)

|             |                                         |
| :---------: | --------------------------------------- |
|  **기간**   | 2019.06 ~ 2019.08                       |
|  **역할**   | 라이브러리 개발 / 크롬 익스텐션 개발    |
| **팀 구성** | 2명                                     |
| **Github**  | <https://github.com/0-Auth/0Auth>       |
|  **비고**   | 2020 공개SW 개발자대회 출품 (동상 수상) |

#### 상세설명

* Node, Typescript, Crypto-js, Elliptic, React, Material-UI
* 전자서명을 활용해 서버에 데이터 저장 없이 사용자 인증을 수행하는 라이브러리.
* 전자서명과 암호화 모듈을 활용한 사용자 인증 라이브러리 개발 및 크롬 확장 프로그램 UI 개발 작업 수행.

## SoundsHub

|             |                                                      |
| :---------: | ---------------------------------------------------- |
|  **기간**   | 2019.06 ~ 2019.08                                    |
|  **역할**   | 팀장 / 백엔드 개발                                   |
| **팀 구성** | 3명                                                  |
| **Github**  | <https://github.com/cnu-bottomup-3m/Team_3m_Projcet> |
|  **비고**   | 2019 교내 프로젝트 경진대회 출품 (대상 수상)         |

#### 상세설명

* HTML, CSS, Javascript, PHP, MySQL, Python, YouTube Data API
* 무료 음악 스트리밍 웹사이트
* NCP 무료 우분투 서버를 기반으로 서버 및 DB 환경 구축.
* BeautifulSoup을 활용해 Melon 차트, YouTube Video id를 크롤링하고 Crontab을 통해 자동화.
* 19년 8월 배포, 무료서버 만료 이후 종료.

<br/>

<div align="center" class="end">

_읽어주셔서 감사합니다._

</br>

---

<br/>

<sub><sup>Frontend Engineer, <a href="https://github.com/anthonyminyungi">@Anthony Min</a></sup></sub>

</div>
