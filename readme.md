
## Woodruff papers python scripture matching repo


### Scriture matches
- [Top standard works matches](data/matches/top_matches.csv)

### Original Data
- [Raw journal entries](data/raw/data_woodruff_raw.csv)
<!-- - [Book of Mormon Matches](top_matches/Book%20of%20Mormon.csv) -->
<!-- - [Doctrine and Covenants Matches](./top_matches/Doctrine%20and%20Covenants.csv) -->
<!-- - [New Testament Matches](./top_matches/top_matches_New%20Testament.csv) -->
<!-- - [Pearl of Great Price Matches](./top_matches/Pearl%20of%20Great%20Price.csv) -->

### Code
- [Python matching script](scripts/scripture_matching.py) uses [IFIDF Vectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
- [DataUtil](DataUtil.py) python utility class with a few data wrangling and general python utility functions.
