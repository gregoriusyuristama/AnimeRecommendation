# Laporan Proyek Machine Learning - Gregorius Yuristama

## Project Overview

Di era modern ini sangat mudah sekali untuk menikmati hiburan seperti film atau serial TV. Banyak penyedia layanan hiburan *streaming* yang memiliki pertumbuhan sangat pesat seperti *Netflix, Disney+, HBO Go*, dll akibat tingginya minat akan hiburan. Media hiburan tersebut menyediakan berbagai hiburan berupa film dan serial TV yang dapat langsung dinikmati melalui *gadget* seperti *smartphone*, televisi, *tablet*, dll. Film dan serial yang disediakan pun berasal dari berbagai macam kategori seperti drama korea, film *hollywood*, serial dokumenter, dan juga *anime*.

[Anime](https://katadata.co.id/intan/berita/62d982116d45a/sejarah-dan-pengertian-anime-kartun-buatan-jepang-yang-populer) adalah animasi buatan Jepang, yang kini berkembang menjadi bentuk budaya populer, dengan basis penggemar yang cukup banyak. Anime tidak hanya dipandang sebagai sarana hiburan semata, sebagian besar orang bahkan menganggap kartun tersebut sebagai tontonan hari-hari, lantaran cerita kerap menginspirasi. Dilansir dari [kumparan](https://kumparan.com/info-anime/ranking-10-besar-negara-pecinta-anime-selama-satu-dekade-versi-google-trends-1t91MqCl8eT/full), Indonesia merupakan pengakses anime terbesar nomor 5 sedunia. Dari hal ini dapat dilihat bahwa *anime* memiliki peminat yang besar di Indonesia dan merupakan salah satu pilihan tontonan untuk sebagian besar orang Indonesia.

Karena hal inilah permasalahan yang akan disesesaikan dalam proyek ini adalah bagaimana memberikan rekomendasi tontonan dengan kategori anime yang sesuai dengan preferensi penonton. Salah satu cara untuk menyelesaikan permasalah ini adalah menggunakan algoritma *machine learning* salah satunya adalah sistem rekomendasi dengan 2 metode *content based recommendation* dan *collaborative filtering*. Dengan menggunakan metode ini diharapkan penonton akan bisa mendapatkan rekomendasi tontonan yang sesuai dengan preferensi mereka sehingga ketika menonton *anime* melalui layanan *streaming* akan terasa lebih nyaman.


## Business Understanding


### Problem Statements

Bagaimana cara untuk memberikan rekomendasi *anime* berdasarkan preferensi tontonan pengguna?

### Goals
Dengan menggunakan salah satu cabang *machine learning* yaitu sistem rekomendasi berdasar data preferensi tontonan pengguna di masa lalu


### Solution Statements

Menggunakan 2 metode dalam sistem rekomendasi, yaitu :

* ***Content Based Filtering*** : Pemfilteran berbasis konten di mana sistem ini memberikan rekomendasi untuk menebak apa yang disukai pengguna berdasarkan aktivitas pengguna tersebut. Pemfilteran berbasis konten menggunakan kesamaan dalam produk, layanan, atau fitur konten, serta akumulasi informasi tentang pengguna untuk membuat rekomendasi.

* ***Collaborative Filtering*** : Proses filter secara kolaboratif adalah teknik yang dapat memfilter item yang mungkin disukai pengguna berdasarkan reaksi dari pengguna serupa.

## Data Understanding

Data yang digunakan dalam membuat model *machine learning* adalah *dataset* berisi id pengguna, *anime* yang ditonton, dan *rating* yang diberikan pengguna kepada *anime* yang berkaitan. Untuk *dataset* yang digunakan dapat diunduh atau dilihat di [sini](https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database)

Data berisi informasi preferensi pengguna dari 73.516 pengguna dan 12.294 judul *anime*. *Dataset* diambil dari website [myanimelist.net](myanimelist.net) menggunakan API.

### Berikut adalah penjelasan tiap kolom untuk seluruh data

Anime.csv

* anime_id - id setiap judul *anime* yang berupa angka.
* name - judul *anime*.
* genre - *genre* atau aliran dari *anime*.
* type - tipe *anime* (Film, Serial, OVA, dll).
* episodes - jumlah episode dari anime (1 jika bertipe film).
* rating - *rating* dalam skala 10 dari *anime*.
* members - *jumlah* anggota komunitas dari anime


Rating.csv

* user_id - id pengguna.
* anime_id - id *anime* yang diberi *rating* pengguna.
* rating - *rating* yang diberikan pengguna dalam skala 10 (-1 jika hanya menonton dan tidak memberi *rating*).

### Exploratory Data Analysis

![Screenshot 2023-01-26 at 11 40 01 AM](https://user-images.githubusercontent.com/102383943/214759754-12453df0-bcc5-45a9-89ac-a5e55cb52110.png)

Gambar 1. Grafik persebaran tipe *anime*

Dari Gambar 1 dapat dilihat bahwa ternyata pada *dataset* ini terdapat *anime* dengan beberapa macam tipe seperti TV/Serial, *OVA*, *Movie*, dll.


![Screenshot 2023-01-26 at 11 39 38 AM](https://user-images.githubusercontent.com/102383943/214759712-13b70728-40eb-4f3f-9466-223b7da5c574.png)

Gambar 2. Grafik persebaran *rating*

Pada Gambar 2 dapat dlihat bahwa rata-rata *rating* *anime* berada diantara 5 sampai 8, dan sangat sedikit sekali *rating* *anime* yang berada dibawah 3 ataupun sempurna yaitu 10.

![Screenshot 2023-01-26 at 11 40 30 AM](https://user-images.githubusercontent.com/102383943/214759803-27061135-59de-49c9-accb-0e2ca614de91.png)

Gambar 3. Grafik *genre* untuk seluruh *anime*

Dari Gambar 3 dapat dilihat bahwa *genre* yang dimiliki *anime* cukup bermacam-macam dan *genre* terbanyak adalah *comedy*

Dari ketiga grafik diatas dapat dilihat bahwa data pada *dataset* ternyata tipe *anime* cukup bermacam-macam. Untuk penelitian kali ini akan difokuskan saja ke *anime* dengan tipe TV/Serial saja karena tipe tersebut merupakan tipe yang paling banyak pada *dataset* ini dan juga kebanyakan orang lebih sering menonton *anime* dengan tipe TV/Serial dan jarang yang menonton tipe film atau lainnya.


![Screenshot 2023-01-26 at 11 40 47 AM](https://user-images.githubusercontent.com/102383943/214759839-0cf40cf8-284d-4209-a5cc-a63d160373ad.png)

Gambar 4. Grafik *genre* untuk *anime* bertipe TV/Serial

Setelah menghapus data *anime* dengan tipe selain TV/Serial dapat dilihat bahwa grafik *genre* pada Gambar 4 tidak terlalu terpengaruh dan datanya masih relatif sama dengan data sebelumnya. Sehingga, data masih baik untuk dilanjutkan ke proses berikutnya.

## Data Preparation

Terdapat 2 data preparation yang dilakukan untnuk setiap metode pada proyek ini : 

* ***Content Based Filtering*** : 

1. Mengambil 2 kolom yang diperlukan yaitu *name* dan *genre*, karena untuk melihat kesamaan antara satu *anime* dengan *anime* lainnya berdasar *genre* maka 2 kolom yang diperlukan untuk *content based filtering* hanyalah kedua kolom tersebut.
2. Melakukan Fitting TF-IDF untuk kolom *genre*, tujuan dari tahapan ini adalah untuk mengukur fitur penting pada setiap anime.

* ***Collaborative Filtering*** :

1. Melakukan *encoding* user_id, dengan tujuan agar setiap user\_id memiliki index masing-masing 
2. Melakukan *encoding* anime_id, dengan tujuan agar setiap anime\_id memiliki index masing-masing 
3. Mengacak *dataset*, agar data yang selanjutnya akan dimasukkan ke *train-test-split* dapat teracak dengan baik dan diharapkan hasil prediksinya pun lebih baik dengan data yang teracak.
4. Melakukan *train-test-split*, data yang sudah diacak kemudian dibagi 90% untuk data *training* dan 10% untuk data *validation*

## Modeling

### *Content Based Filtering*

Kelebihan : 

* Dapat mengambil keputusan rekomendasi berdasarkan karakteristik dari *item* yang direkomendasikan, seperti kategori, deskripsi, atau kata kunci.

* Rekomendasi yang diberikan sesuai dengan preferensi pengguna, karena berdasarkan karakteristik *item* yang diterima oleh pengguna sebelumnya.

* Bisa menangani masalah *cold start* dengan mudah, karena tidak tergantung pada data interaksi pengguna sebelumnya.

Kekurangan :

* Menghasilkan rekomendasi yang terlalu mirip dengan item yang telah diterima oleh pengguna sebelumnya, sehingga kurang variatif.

* Kurang efektif jika karakteristik item tidak representatif dari item tersebut atau jika preferensi pengguna berubah.

* Dapat mengalami masalah *overfitting* jika tidak diterapkan dengan benar.

Pada Proyek ini terdapat beberapa tahapan yang dilakukan untuk melakukan *modeling* dengan metode *content based filtering*, yaitu : 

1. **Membuat matriks _cosine similarity_** : Pada tahapan ini, kemiripan antara satu *anime* dengan *anime* lainnya direpresentasikan dalam matrix dengan perhitungan *cosine similiarity* berdasarkan *fitting* TF-IDF pada *data preparation*

TF-IDF adalah *term frequency-inverse document frequency* yang dapat diartikan seberapa penting sebuah fitur dalam kumpulan data. Perhitungan dari TF-IDF adalah : 

$$w_{i,j} = tf_{i,j} \cdot  log(\frac{N}{df_{i}})$$

Keterangan : 

tfi,j = jumlah kemunculan i pada dokumen ke-j

dfi = jumlah dokumen

N = total jumlah dokumen

Sehingga pengaplikasiannya pada proyek ini adalah tfi,j adalah jumlah kemunculan *genre* i dari sebuah *anime*. Lalu df\_i adalah jumlah *genre* i yang muncul pada seluruh *anime* dan N adalah jumlah *anime* keseluruhan. Sehingga semakin sedikit kemunculan *genre* (df_i) maka semakin besar pula bobot *genre* tersebut.

Lalu hasil dari matrix TF-IDF tersebut kemudian dihitung *cosine similarity*-nya dengan rumus:


$$ cosine(v,w) = \frac{v\cdot w}{\left | v \right | \left | w \right |}=  \frac{\displaystyle\sum_{i=1}^{N} v_{i}w_{i}}{\sqrt{\displaystyle\sum_{i=1}^{N}v_{i}^{2}}\sqrt{\displaystyle\sum_{i=1}^{N}w_{i}^{2}}}  $$

Sebagai contoh perhitungan digunakan tabel berikut : 

Tabel 1. contoh matrix berdasarkan *dataset*

| name                        | fantasy  | game | school   | action   |
|-----------------------------|----------|------|----------|----------|
| Choegang Habche: Mix Master | 0.738211 | 0    | 0        | 0        |
| Chii Jiaan Chuaanqii        | 0.623870 | 0    | 0        | 0.534590 |
| Azumanga Daioh              | 0        | 0    | 0.655717 |0 |

Perhitungan cosine similarity untuk ketiga *anime* adalah : 

$$ cosine(Choegang, Chii) =  \frac{0.738 \times 0.623 + 0 \times 0 + 0 \times 0 + 0 \times 0.534}{\sqrt{0.738^2 + 0^2 + 0^2 + 0^2}\sqrt{0.623^2 + 0^2 + 0^2 + 0.534^2}} = \frac{0.46 + 0 + 0 + 0}{\sqrt{0.545 + 0 + 0 + 0}\sqrt{0.390 + 0 + 0 + 0.286}}  = \frac{0.46}{\sqrt{0.831}} = 0.504 $$

$$ cosine(Chii, Azumanga) = \frac{0.623 \times 0 + 0 \times 0 + 0 \times 0.655 + 0.53 \times 0}{\sqrt{0.623^2 + 0^2+0^2+0.534^2}\sqrt{0^2+0^2+0.655^2+0^2}} = \frac{0 + 0 + 0+ 0}{\sqrt{0.388+0+0+0.285}\sqrt{0+0+0.429+0}} = \frac{0}{\sqrt{1.102}} = 0 $$

$$ cosine(Choegang, Azumanga) = \frac{0.738 \times 0 + 0 \times 0 + 0 \times 0.655 + 0 \times 0}{\sqrt{0.738^2+0^2+0^2+0^2}\sqrt{0^2+0^2+0.655^2+0^2}}= \frac{0 + 0 + 0 + 0}{\sqrt{0.544} \sqrt{0.429}} = \frac{0}{\sqrt{0.973}} = 0 $$

Dari perhitungan diatas dapat dilihat bahwa *similarity* (kemiripan) untuk *anime* Choegang Habche: Mix Master dengan Chii Jiaan Chuaanqii adalah 0.504, lalu *anime* Chii Jiaan Chuaanqii dengan Azumanga Daioh adalah 0, dan yang terakhir *anime* Choegang Habche: Mix Master dengan Azumanga Daioh adalah 0. Semakin besar nilai *similarity*-nya maka semakin mirip pula antara satu *anime* dengan *anime* lainnya.


2. **Mendapatkan Rekomendasi**: Hasil rekomendasi dari penggunaan *content based filtering* pada proyek ini.


Tabel 2. Hasil Top-10 Rekomendasi Anime untuk *Naruto*

| Nama Anime  |Genre   |
|---|---|
|Naruto: Shippuuden|Action, Comedy, Martial Arts, Shounen, Super Power|
|Rekka no Honoo |Action, Adventure, Martial Arts, Shounen, Super Power|
|Kurokami The Animation|Action, Martial Arts, Super Power|
|Project ARMS |Action, Martial Arts, Super Power|
|Wolverine | Action, Martial Arts, Super Power|
|Dragon Ball Z | Action, Adventure, Comedy, Fantasy, Martial Arts, Shounen, Super Power |
|Dragon Ball Kai (2014) | Action, Adventure, Comedy, Fantasy, Martial Arts, Shounen, Super Power|
|Dragon Ball Kai | Action, Adventure, Comedy, Fantasy, Martial Arts, Shounen, Super Power|
|Dragon Ball Super |Action, Adventure, Comedy, Fantasy, Martial Arts, Shounen, Super Power
|Dragon Ball |Adventure, Comedy, Fantasy, Martial Arts, Shounen, Super Power|


Dari hasil rekomendasi pada Tabel 2 dapat dilihat bahwa hasil yang diberikan cukup baik karena ketika mencarikan rekomendasi untuk *anime* Naruto dengan *genre* *Action, Comedy, Martial Arts, Shounen, Super Power* hasil yang diberikan juga memiliki *genre* yang mirip. Seperti pada Tabel 2, rekomendasi pertama adalah anime Naruto: Shippuuden yang memiliki *genre* sama persis diikuti Rekka no Honoo yang memiliki *genre* yang mirip hanya berbeda di *genre* *Comedy* di Naruto dengan *Adventure* pada Rekka no Honoo.

### *Collaborative Filtering*

Kelebihan : 

* Tidak memerlukan informasi tentang item yang direkomendasikan, hanya memerlukan data interaksi pengguna.

* Dapat menangkap preferensi pengguna yang tidak terungkap secara eksplisit.

* Dapat menangkap hubungan antar item yang tidak jelas.

Kekurangan: 

* Memerlukan data interaksi pengguna yang cukup banyak untuk menghasilkan rekomendasi yang baik.

* Dapat menghasilkan rekomendasi yang bias terhadap *item* populer.

* Dapat menghasilkan rekomendasi yang kurang relevan untuk pengguna baru atau pengguna yang tidak memiliki interaksi yang cukup dengan sistem.

Pada Proyek ini tahapan yang dilakukan untuk melakukan *modeling* pada *collaborative filtering* adalah sebagai berikut : 

1. **Membuat kelas RecommenderNet** : Untuk kelas RecommenderNet pada proyek ini terinspirasi dari situs [Keras](https://keras.io/examples/structured_data/collaborative_filtering_movielens/) dengan melakukan beberapa adaptasi untuk disesuaikan dengan kasus pada proyek ini. 

Pada kelas ini merupakan sebuah *model* dengan beberapa *layer*. 

* *Layer* pertama adalah *layer Embedding* untuk pengguna yang memiliki masukan sebesar jumlah penggunda dan besar keluaran sesuai *input* *embedding*  pada parameter. 
* Lalu *layer* kedua adalah untuk *user bias* berupa *layer embedding* yang memiliki inputan sebesar jumlah pengguna dan besar keluaran senilai 1.
* *Layer* ketiga adalah *layer Embedding* untuk *anime* dengan besar masukan sebesar banyak *anime* dan besar *embedding*
* *Layer* keempat adalah *anime bias* berupa *layer embedding* yang memiliki masukan sebesar jumlah *anime* dan keluaran 1
* Lalu yang kelima merupakan sebuah perkalian *dot* untuk *layer embedding* pengguna (*layer* pertama) dengan *layer embedding anime* (*layer* ketiga)
* Kemudian nilai x adalah jumlah dari *user bias* (*layer* kedua) ditambah dengan *anime_bias* (*layer* keempat) dan hasil perkalian *dot* pada poin 5.
* Terakhir, dilakukan *aktivasi sigmoid* untuk nilai x agar nilai yang dihasilkan berada diatara 0 sampai 1.

2. **Melakukan Proses _training_** : Proses training dilakukan dengan parameter *batch_size* 256, dan *epoch* sebanyak 15

3. **Mendapatkan Rekomendasi** : Hasil rekomendasi dari penggunaan *collaborative filtering* pada proyek ini.

![Screenshot 2023-01-26 at 2 04 55 PM](https://user-images.githubusercontent.com/102383943/214776303-f1d60342-a611-411a-83c9-90490560668f.png)

Gambar 6. Rekomendasi Top-10 *anime* untuk pengguna 34899 menggunakan *collaborative filtering*

Dari Gambar 6 hasil rekomendasi menggunakan *collaborative filtering* cukup baik, karena dari pengguna 34899 cukup sering menonton *anime* dengan *genre* *Action, Adventure, Comedy*, dan *Drama* akan direkomendasikan dengan *anime* dengan *genre* serupa.


## Evaluation

1. **Hasil Evaluasi _Content Based Filtering_**

Untuk mengevaluasi *Content Based Filtering* digunakan metrik *precision* dengan perhitungan : 

$$ Precision = \frac{\textrm{number of our reccomendation that are relevant}}{ \textrm{number of items we recommended}}$$

Hasil dari rekomendasi menggunakan *Content Based Filtering* dapat dilihat pada Tabel 2. Berdasarkan gambar tersebut, dapat dilihat untuk rekomendasi *anime* Naruto dengan 10 rekomendasi. Dari 10 rekomendasi tersebut *anime* yang memiliki *genre* sama persis dengan Naruto berjumlah 6 yaitu Naruto: Shipuudden dan seluruh rekomendasi Dragon Ball yang berjumlah 5. Sehingga nilai presisinya adalah 6/10 atau 60%.

2. **Hasil Evaluasi _Collaborative Filtering_**

Untuk mengevaluasi *Collaborative Filtering* metrik yang digunakan adalah *Root Mean Squared Error* atau RMSE dengan rumus sebagai berikut : 

$$ RMSE = \sqrt{\frac{1}{N}\displaystyle\sum_{i=1}^{N}(y_{i} - y\textunderscore pred_{i})^2} $$

Keterangan: 

*N = jumlah dataset*

*y<sub>i</sub> = nilai sebenarnya*

*y\_pred = nilai prediksi*

Dari rumus diatas, dapat dilihat bahwa hasil prediksi dapat dikatakan baik jika nilai dari prediksi tidak jauh dari sebenarnya. Sehingga hasil rekomendasi yang baik adalah yang memiliki nilai RMSE yang kecil. 

![Screenshot 2023-01-26 at 2 04 30 PM](https://user-images.githubusercontent.com/102383943/214776248-5c358214-e02b-475a-999b-c8f27508b7f8.png)

Gambar 7. Hasil visualisasi *training* dengan metrik yang dinilai adalah *root mean squared error* (RMSE)

Hasil dari *training* untuk *collaborative filtering* menggunakan *tensorflow* sudah cukup baik, dapat dilihat pada grafik Gambar 7, nilai dari RMSE semakin turun sejak awal pelatihan hingga *train* dan *validation*-nya konvergen setelah *epoch* 8. Nilai RMSE pada pelatihan berakhir adalah 0.1316 pada *train* dan 0.1331 pada *validation*. Hasil ini merupakan hasil yang cukup baik karena semakin kecil nilai RMSE, maka semakin baik pula sistem rekomendasi tersebut. 

Dapat dilihat juga pada hasil rekomendasi pada Gambar 6, dapat dilihat bahwa pengguna nomor 34899 yang sering menonton *anime* dengan *genre* *Action, Adventure, Comedy* dan *Drama* direkomendasikan juga dengan *anime* yang memiliki satu atau lebih dari *genre* tersebut.



## Conclusion

Dari hasil penelitian ini dapat dilihat bahwa sistem rekomendasi *anime* dengan menggunakan *content based filtering* dan *collaborative filtering* dapat memberikan rekomendasi yang sesuai dengan kelebihan dan kekurangan dari masing-masing metode. *Content Based Filtering* akan lebih efektif diaplikasikan kepada pengguna yang menyukai *anime* dengan *genre* tertentu saja, sedangkan *Collaborative Filtering* akan lebih efektif diaplikasikan ke pengguna yang memiliki kemiripan besar dengan pengguna lain. Untuk penelitian selanjutnya dapat dilakukan improvisasi untuk *anime* dengan seluruh tipe, tidak terbatas pada tipe TV/Serial saja serta memperhitungkan tipe *anime* sebagai preferensi pengguna.
