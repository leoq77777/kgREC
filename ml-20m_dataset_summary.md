# MovieLens 20M Dataset Summary

## Overview
The MovieLens 20M dataset contains 20000263 ratings and 465564 tag applications across 27278 movies, created by 138493 users between January 09, 1995 and March 31, 2015. It was generated on March 31, 2015 and updated on October 17, 2016.

## File Structure
The dataset consists of six files:
- `genome-scores.csv`
- `genome-tags.csv`
- `links.csv`
- `movies.csv`
- `ratings.csv`
- `tags.csv`

## Data File Details

### ratings.csv
Contains 20000264 lines of user-movie ratings with format:
`userId,movieId,rating,timestamp`
- Ratings are on 5-star scale with half-star increments
- Timestamps are seconds since UTC of January 1, 1970
- Ordered by userId then movieId

### movies.csv
Contains 27279 lines with movie metadata:
`movieId,title,genres`
- Movie titles include release year
- Genres are pipe-separated categories
- Only includes movies with at least one rating or tag

### tags.csv
Contains 465565 lines of user-generated tags:
`userId,movieId,tag,timestamp`
- Tags are free-text user-generated content
- Timestamps follow the same format as ratings

### genome-scores.csv & genome-tags.csv
- `genome-tags.csv`: 1129 lines mapping tag IDs to tag text
- `genome-scores.csv`: 11709769 lines with relevance scores (0-1) for each movie-tag pair

### links.csv
Contains 27279 lines with external movie IDs:
`movieId,imdbId,tmdbId`
- Maps MovieLens IDs to IMDB and TMDB IDs
- Enables integration with external movie databases

## Usage Guidelines
- Dataset may be used for research purposes with proper citation
- Commercial use requires permission from GroupLens Research Project
- Users must acknowledge use in publications
- Redistributing the data requires separate permission

## Citation
When using this dataset, please cite:
> F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4, Article 19 (December 2015), 19 pages.

## Technical Details
- All files are UTF-8 encoded CSV with header rows
- Fields containing commas are escaped with double quotes
- User and movie IDs are consistent across all files
- Movie IDs correspond to those used on movielens.org