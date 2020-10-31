# NLP 분석 실습파일과 주요 내용 (9번, 10번 파일만 Local PC에서 실행가능, 나머지는 Colab에서 실행가능)

# 2_IDF.ipynb
 - Python을 활용하여 TF 계산
 - Python의 활용하여 IDF 계산
 - sklearn을 활용하여 TF, DF, TF-IDF 계산
 - 모델기반의 클러스터링을 위한 k-Means 적용예제
 
# 3_SimpleCrawling.ipynb
 - wikipedia 텍스트 데이터 수집
 - 웹페이지 크롤링: Request 라이브러리 활용, BeautifulSoup 라이브러리 활용
 
# 4_MeCab.ipynb
 - Colab에서의 MeCab 설치
 
# 4_Pre_Processing.ipynb
 - NLTK를 활용한 자연어처리
 - 단어 토큰화 - nltk의 정규표현식 사용
  . \w+ 는 문자 또는 숫자가 1개 이상인 경우를 인식, 단어들 중심으로 토큰화 수행, 문장부호는 생략됨(ex. (,), j)
  . 문장의 부호가 생략되면 해당 의미가 손상되는 경우도 있음 (ex, 5-9 등)
 - TextBlob 사용한 토큰화
  . words 함수: 단어토큰화, sentences: 문장토큰화
  . sentiment: -1~1사이의 극성(polarity)과 주관성(Subjectivity) 표현(0는 객관적, 1은 주관적)
 - Keras를 사용한 토큰화
  . text_to_word_sequence(msg,filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=' '))
 - NLTK n-gram 예제
 - n-gram 함수
 - HTML Tag 제거
 - 소문자 변환처리
 - 정규표현식을 활용: 숫자과 전화번호 치환, 정규표현식을 이용한 이메일, URL, 한글자음/모음, HTML, 문장부호 정제 함수로 적용
 - NLTK 패키지 활용 전처리
  . FreqDist(문서에 사용된 단어(토큰)의 사용빈도 정보를 담는 클래스)
  . 형태소 분석
  . 어간 추출(stemming)
  . 표제어 추출(Lemmatization)
  . NER(Named Entity Recognition)
 - Mecab
 - Hannanum 
 - Kkma
 - Komoran : Komoran(userdic='dic.txt')
 - Okt: print(okt.pos(u'이것도 되나욬ㅋㅋ', norm=True, stem=True))

# 5_OneHotEncode.ipynb
 - scikit-learn을 사용한 One hot Encode
 - Kereas를 사용한 One-hot Encode
 - Sklearn의 CountVectorizer를 활용한 Bag of Words
 - Naive Bayes(스팸분류) 예제: 
 
  <ol>1)Dataset 다운로드 및 정제  
  <br>2)DataSet 탐색적 데이터 분석: messages.groupby('class').describe(), messages.hist(column='length',by='class',bins=50, figsize=(15,6)), 특수기호(문자구분자), 불용어, 소문자 처리 함수  
  <br>3)데이터 집합을 분할하여 기능 훈련 및 테스트 데이터 생성  
  <br>4)사이킷런의 CountVectorizer를 통해 피처 생성  
  <br>5)모델 적용 - PipeLine 적용하여 코드 간략화
  </ol>
  
# 5_Word2Vec_FastText.ipynb
 - Word2Vec
 - FastText
 - word2vec 시각화
 
# 6_Similarity.ipynb
 - 코사인 유사도(Cosine Similarity)
두 벡터 사이의 각도의 코사인을 측정 하여 유사성을 계산
코사인 유사성을 사용하여 문장을 벡터로 변환
1) BoW(TF) : 일반적인 문서 유사도 비교에 좋음
2) TF-IDF : 검색의 목적에는 유용함
3) Word2Vec : 문맥기반의 유사성 비교에 용이

 - Jaccard 유사성
비교 대상 2개 문서의 교집합의 크기를 2개 문서의 합집합 크기로 나눈 것으로 정의

# 6_TopicModeling.ipynb
 - 네이버 영화리뷰 데이터셋 활용
 - 토픽모델 구축(LDA Model 학습)
 
corpus : 문서 벡터
id2word : 단어ID와 매핑된 단어의 빈도
num_topics : 가설 토픽 개수
chunksize : 훈련 알고리즘에 사용되는 문서사이즈, 빠른 학습을 위해서 사이즈 상향
Hoffman의 논문에 의하면 Chunksize는 모델 품질에 영향은 일부 있음
pass : epochs와 같은 용어로 전체 코퍼스에서 모델을 학습시키는 빈도
iteration : 각각 문서에 대한 반복횟수, passes & iteration 은 많을수록 좋음
per_word_topics : 각 단어에 대해 가장 가능성이 높은 주제의 내림차순으로 토픽목록 x 피처길이 = 토픽값 계산

# 7_ngram.ipynb
 - 1) NLTK의 N-gram 적용
 - 2) TextBlob를 사용한 N-gram
 - 4) 한글 형태소 분석기를 사용한 ngram
 - 6) 조건부 확률의 추정
 
# 9-1_RNN.ipynb --> Local PC 실행
 - hihello에서 마지막 o를 추측하는 예제
# 9-2_LSTM.ipynb --> Local PC 실행
 - hihello에서 마지막 o를 추측하는 예제
# 9-3_GRU.ipynb --> Local PC 실행
 - hihello에서 마지막 o를 추측하는 예제
# 9-4_BLSTM.ipynb --> Local PC 실행
 - MNIST image에서 image가 0에서 9중 어떤 숫자인지 예측하는 Bidirectional LSTM
 
# 10_Seq2Seq.ipynb --> Local PC 실행
 - 짧은 영어 문장을 짧은 프랑스어 문장으로 번역하는 예제
 
# 11_Attention.ipynb
 - Attention을 활용한 번역예시(Sequence to Sequence 적용)
 
# 12_transformer.ipynb
 nn.TransformerEncoder 모델을 언어 모델링(language modeling) 과제에 적용한 예제임

# 13_BERT_Classification.ipynb
 - HuggingFace(BERT 모델 같은 트랜스포머 모델들을 쉽게 다룰 수 있게 해주는 패키지) 활용
 - BERT를 활용한 스팸분석 모델 만들기




 
  
  
  

  
