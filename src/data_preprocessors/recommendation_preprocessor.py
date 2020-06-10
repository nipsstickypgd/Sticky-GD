import csv

from common import data_folder


def recommentation_dataset():
    with open(data_folder + "ratings_big.csv") as fin:
        reader = csv.reader(fin, delimiter=',')
        next(reader)
        ratings = []
        # num = 0
        for row in reader:
            ratings.append((int(row[0]), int(row[1]), float(row[2])))
            # num += 1
            # if num > 20000:
            #     break

        user_cnt = 1 + max(int(row[0]) for row in ratings)
        movie_cnt = 1 + max(int(row[1]) for row in ratings)
        print('Read completed.', 'Users:', user_cnt, 'Movies:', movie_cnt, 'Ratings:', len(ratings))

        max_user_cnt = 20000
        max_movie_cnt = 20000
        filtered_ratings = [rating for rating in ratings if rating[0] < max_user_cnt and rating[1] < max_movie_cnt]
        user_cnt = 1 + max(int(row[0]) for row in filtered_ratings)
        movie_cnt = 1 + max(int(row[1]) for row in filtered_ratings)
        print('Filtering completed.', 'Users:', user_cnt, 'Movies:', movie_cnt, 'Ratings:', len(filtered_ratings))
        movie_ratings = [0] * movie_cnt
        cnt = [0] * movie_cnt
        for _, movie, rating in filtered_ratings:
            cnt[movie] += 1
            movie_ratings[movie] += rating

        avg_ratings = []
        for movie in range(movie_cnt):
            if cnt[movie] == 0:
                avg_ratings.append(3)
            else:
                avg_ratings.append(movie_ratings[movie] / cnt[movie])

        return filtered_ratings, user_cnt, movie_cnt, avg_ratings
