#CSCI 566 Spring 24 - Group Project

Understanding
data to be collected
1. tweet content (t-1)
2. tweet content (t)
3. reddit content (t-1)
4. #likes
5. #retweets
6. #comments
7. timestamp t-1
8. timestamp t
--
prep
1. 2. -> embeddings (prep) , sentiment pred, sector pred
3 -> reddit sentiment (this has to be an aggregate of all textual content pushed through to pred sentiment)
--
 input
1. tweet emb t
2. tweet emb t-1
3. tweet sentiment t
4. tweet sentiment t-1
5. reddit sentiment t-1
6. # likes t-1
7. #retweets t-1
8. #comments t-1
9. #sec t-1
10. #sec t
11. timestamp t-1
12. timestamp t
