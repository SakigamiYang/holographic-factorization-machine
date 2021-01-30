# Holographic Factorization Machines

An implementation for [Holographic Factorization Machines (HFM)](https://ojs.aaai.org/index.php/AAAI/article/view/4448).

As an example, trained with data 'rating.csv' of [Anime Recommendations Database](https://www.kaggle.com/CooperUnion/anime-recommendations-database) downloading from kaggle.

Data format:

```text
user_id, anime_id, rating
```

Only used feature engineering processes as:
1. Turn user_id from 1\~M into 0\~M-1
2. Turn anime_id from 1\~N into 0\~N-1
3. Turn rating from 1\~10 into 0.1\~1.0 by divided by 10.
