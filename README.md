此課程是8個真實案件(專案)，利用機器學習分析作資料分析，我會將在每個專案中學習到的關鍵套件(程式碼)和之前較少接觸的演算法理論整理。
【Case1】
這是開發一個模型來預測客戶願意支付的總金額，經過資料的探索來建立模型。

這邊學習到可以利用model.summary()，打印出模型的摘要信息，包括每一層的結構、參數數量等，可以將這個打印出來在往後調整參數或與團隊溝通都是非常重要的。
model = Sequential()
model.add(Dense(25, input_dim=5, activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Dense(1, activation='linear'))
model.summary()

【Case2】
利用python資料庫裏面的圖片進行影像辨識，主要分類成10種動物，根據deep learning進行預測分類。

這邊學習到可以將model暫存起來，不用每次都重跑結果，只要重新呼叫即可。
# save the model
directory = os.path.join(os.getcwd(), 'saved_models')

if not os.path.isdir(directory):
    os.makedirs(directory)
model_path = os.path.join(directory, 'keras_cifar10_trained_model_Augmentation.h5')
cnn_model.save(model_path)


【Case3 & Case4】
Case3 & Case4這兩個專案，主要都是利用FaceBook所開發的套件fbprophet(https://facebook.github.io/prophet/docs/quick_start.html)，
主要適用於週期性的數值變化，輸入時間與數值兩個欄位就會預估未來數值的變化。NOTE:目前這個套件需要改成from prophet import Prophet。

Case3是預測芝加哥的犯罪率，內容是芝加哥犯罪資料集包含 2001 年至 2017 年芝加哥市報告的犯罪事件摘要。(資料集來自芝加哥警察局的 CLEAR（公民執法分析和報告）系統)
Case4 是酪梨零售的銷售數據代表 2018 年全國零售量（單位）和價格的每週零售掃描數據。( 零售掃描資料直接來自零售商的收銀機，基於哈斯酪梨的實際零售銷售)

裡面程式碼.resample('Y')：這個部分是對資料進行重新取樣的操作，其中的 'Y' 表示年度（Yearly）的重新取樣。這意味著程式將按照年份對資料進行分組，改”M”就變成以月去呈現，”Q”以季呈現。

plt.plot(chicago_df.resample('Y').size()) #.resample('Y')：這個部分是對資料進行重新取樣的操作，其中的 'Y' 表示年度（Yearly）的重新取樣。這意味著程式將按照年份對資料進行分組。





【Case5】
使用 KERAS 中的 LE-NET 架構對21種交通標誌進行分類的程式碼，屬於影像辨識。

這邊學習到可以對資料集進行洗牌（shuffle），可以使用 sklearn.utils.shuffle 函數。這個函數可以將特徵和標籤列表一起洗牌，確保特徵和對應的標籤仍然對應。這對於在訓練模型之前將資料集隨機化很有用，以確保模型在訓練過程中不受到資料排列順序的影響。
## Shuffle the dataset  打亂資料
from sklearn.utils import shuffle
X_train, y_train = shuffle(X_train, y_train)
之前學習到的評斷模型好壞的方式，可能用準確率，召回率等等，此專案提供另外一種用視覺化評斷模型好壞，就是熱度圖（Heatmap），利用sns.heatmap(cm, annot=True)：這行程式碼使用 Seaborn 庫中的熱度圖（Heatmap）功能，將混淆矩陣以視覺化的方式呈現在圖表中。參數 annot=True 表示在熱度圖中顯示數字標籤，以表示混淆矩陣的各個元素。
這樣就可以通過熱度圖清楚地看到模型在不同類別間的分類情況，有助於評估模型的性能和錯誤模式。


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true, predicted_classes_N)
plt.figure(figsize = (25,25))
sns.heatmap(cm, annot=True)





【Case6 & Case7】
Case6 和Case7 都是使用樸素貝葉斯演算法偵測文字或內容進行分類，Case6是否為垃圾郵件，Case7是根據「星星」的數量表示客戶給予的業務評級，範圍從 1 到 5；兩個專案使用的詞頻分類和演算法一樣，故一起討論。

這個是這個常量通常用於文本處理中，可以方便地檢測文本中是否包含標點符號，或者在文本處理過程中去除標點符號。例如，你可以使用 string.punctuation 來過濾文本中的標點符號，或者在拆分句子時將標點符號視為分隔符。

import string
string.punctuation
其內容為   !"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~

這也是基本的stopwords 套件，在NLP可以剃除
# You have to download stopwords Package to execute this command
from nltk.corpus import stopwords
stopwords.words('english')


對於文本分析為什麼要使用 fit_transform？
1.fit_transform 的過程中，CountVectorizer 會根據提供的文本數據來構建詞彙表和詞頻矩陣，這些信息將用於後續的特徵表示。
2.使用 fit_transform 能夠保證特徵提取器和數據的一致性，確保特徵的映射關係和文檔的對應關係是正確的。
3.fit_transform 通常在訓練模型之前用於將原始數據轉換為機器學習算法可以處理的特徵表示形式。
總之，fit_transform 是特徵提取器在訓練階段中常用的方法，它將原始數據轉換為可以被機器學習模型處理的特徵表示形式。
from sklearn.feature_extraction.text import CountVectorizer
sample_data = ['This is the first document.','This document is the second document.','And this is the third one.','Is this the first document?']

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(sample_data)

classification_report 函數是用於生成模型的分類報告，提供了模型在每個類別上的主要評估指標。該報告通常包括精確率（Precision）、召回率（Recall）、F1 值（F1-Score）和支持數（Support）等指標，這些指標可以幫助你評估模型的性能。
這些指標是用於評估分類模型性能的重要指標，它們可以幫助你了解模型在不同類別上的預測表現。下面是這些指標的解釋：

1.精確率（Precision）： 精確率衡量的是模型在預測為正類的樣本中，有多少是真正的正類。它計算方式為 True Positives / (True Positives + False Positives)，其中 True Positives 表示模型正確預測的正類樣本數量，False Positives 表示模型錯誤地將負類預測為正類的樣本數量。精確率越高，表示模型在預測正類時的準確性越高。

2.召回率（Recall）： 召回率衡量的是模型在所有實際正類樣本中，有多少被成功預測為正類。它計算方式為 True Positives / (True Positives + False Negatives)，其中 True Positives 表示模型正確預測的正類樣本數量，False Negatives 表示模型錯誤地將正類預測為負類的樣本數量。召回率越高，表示模型對於正類的覆蓋率越高。

3.F1 值（F1-Score）： F1 值是精確率和召回率的調和平均數，它綜合考慮了模型的準確性和覆蓋率。F1 值的計算方式為 2 * (Precision * Recall) / (Precision + Recall)。F1 值越高，表示模型在精確率和召回率之間取得了較好的平衡。

4.支持數（Support）： 支持數表示每個類別在測試集中的樣本數量。它是用於計算其他指標的基礎數據，可以幫助你了解每個類別的數據分佈。

總的來說，精確率衡量模型的預測準確性，召回率衡量模型的覆蓋率，F1 值綜合考慮了這兩者，而支持數則提供了類別的數據分佈信息。這些指標通常會結合在一起來全面評估模型的性能。
print(classification_report(y_test, y_predict_test))


其實Naïve bayes 分類器有三種演算法在不同的使用情境下使用。
這三個 Naive Bayes 分類器在特定情況下的使用時機如下：

1.Multinomial Naive Bayes：
特徵類型： 適用於具有離散特徵（例如詞頻、詞袋模型等）的分類任務。
先驗分佈： 預設的先驗分佈為多項式分佈，通常用於處理文本分類等具有離散特徵的問題。
2.Gaussian Naive Bayes：
特徵類型： 適用於具有連續特徵且假設先驗機率為高斯（常態）分佈的分類任務。
先驗分佈： 假設特徵的先驗機率服從高斯分佈，適用於處理連續型數據的分類問題。
3.BernoulliNB Naive Bayes：
特徵類型： 適用於二元特徵（二值型特徵，例如存在/不存在、是/否等）的分類任務。
先驗分佈： 同樣適用於具有離散型特徵，但其特點在於專為二元分類（Binary Classification）而設計，假設先驗機率服從伯努利分佈。

所以選擇哪種 Naive Bayes 分類器取決於你的特徵類型和分類任務的性質。如果你的特徵是離散的，可以考慮使用 Multinomial Naive Bayes 或 BernoulliNB Naive Bayes，而如果特徵是連續的，則使用 Gaussian Naive Bayes。需根據具體情況來選擇最適合的模型。

前面的文本特徵提取轉換這邊介紹了CountVectorizer() 和 fidfTransformer() 都是用於文本特徵提取和轉換的工具，但它們在特徵表示上有所不同，適用於不同的情況。

CountVectorizer(): 這個工具將文本轉換成詞袋模型的特徵向量，它會計算每個詞在文本中出現的頻率，然後用這些頻率作為特徵向量的值。這種方法適用於簡單的文本分類任務，它忽略了詞在文本中的重要性和上下文關係，僅關注詞的出現次數。
from sklearn.feature_extraction.text import CountVectorizer
sample_data = ['This is the first document.','This document is the second document.','And this is the third one.','Is this the first document?']

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(sample_data)

TfidfTransformer(): 這個工具則將文本特徵向量轉換成 TF-IDF 加權的特徵向量，TF-IDF 考慮了詞在文本中的重要性和整個文檔集合中的頻率。TF-IDF 會對常見詞語（如“the”、“is”等）給予較低的權重，而對於罕見詞語給予較高的權重，這有助於更好地表示文本的特徵。
from sklearn.feature_extraction.text import TfidfTransformer

emails_tfidf = TfidfTransformer().fit_transform(spamham_countvectorizer)
print(emails_tfidf.shape)

要判斷哪一種方法更好取決於你的任務和數據集。如果你的任務是簡單的文本分類或者你關心的是詞的出現頻率，那麼使用 CountVectorizer() 可能會更適合。但如果你的任務需要更好地捕捉詞的重要性和上下文關係，那麼使用 TfidfTransformer() 可能會更好。

通常，對於大多數的文本分類和自然語言處理任務，使用 TF-IDF 加權的特徵向量往往會取得更好的效果，因為它更能夠反映詞在文本中的重要性。


【Case8】
此專案是根據大眾影評，針對你給的評價(這邊是用等級區分1-5最喜歡為5)，運用此資料做相關係數，並且媒合資料後推薦給大眾。

此資料前半部分為使用pearson相關係數，並設定評分次數高於80分才會進行相關係數分析。
movie_correlations = userid_movietitle_matrix.corr(method = 'pearson', min_periods = 80) #min_periods = 80 並設定了最小的評分次數為 80（即至少有 80 個評分）才進行相關性計算這樣可以避免因為評分數量太少而導致相關性計算不可靠的情況

第二重要的是這段程式碼的目的是根據你給定的電影評分和相關性計算出與之相似的電影清單。

similar_movies_list = pd.Series()：創建一個空的 Pandas Series，用來存儲相似電影的相關性加權值。
for i in range(0, 2):：遍歷範圍從 0 到 1 的迴圈，這裡假設你給定了 2 部電影的評分。
similar_movie = movie_correlations[myRatings['Movie Name'][i]].dropna()：從相關性矩陣 movie_correlations 中選擇與第 i 部電影相關的電影，並去除其中的缺失值（NaN）。
similar_movie = similar_movie.map(lambda x: x * myRatings['Ratings'][i])：將相似性按照你給定的評分加權，這樣可以得到每部相似電影的加權相似性。
similar_movies_list = similar_movies_list.append(similar_movie)：將每部相似電影的加權相似性添加到 similar_movies_list 中，最終得到了一個包含所有相似電影和對應加權相似性的 Series。

總體來說，這段程式碼的目的是通過計算相關性和評分加權，生成了一個帶有加權相似性的相似電影清單。


import pandas as pd

similar_movies_list = pd.Series()
for i in range(0, 2):
    similar_movie = movie_correlations[myRatings['Movie Name'][i]].dropna() # Get same movies with same ratings
    similar_movie = similar_movie.map(lambda x: x * myRatings['Ratings'][i]) # Scale the similarity by your given ratings
    similar_movies_list = pd.concat([similar_movies_list, similar_movie])


