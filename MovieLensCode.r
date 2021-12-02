##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

if(!require(caret)) install.packages("caret")
if(!require(tidyverse)) install.packages("tidyverse")
if(!require(data.table)) install.packages("data.table")
if(!require (dplyr)) install.packages("dplyr") 
if(!require(sqldf)) install.packages("sqldf")
if(!require(lubridate)) install.packages("lubridate")
if(!require(ggthemes)) install.packages("ggthemes")

library(tidyverse)
library(caret)
library(data.table)
library(dplyr)
library(sqldf)
library(lubridate)
library(ggthemes)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

# Get look-see of dataset structure
str(edx)

# Check features presence in the edx dataset. 
summary(edx)

# Check for missing data
anyNA(edx)


# Get big picture of users and movies in dataset
edx%>% 
  summarize(n_users = n_distinct(userId),
            n_movies = n_distinct(movieId))

# Show in Chart 1 
edx %>% 
  ggplot() +
  geom_histogram(aes(x=edx$rating), bins = 30, 
  fill = "yellow", color="blue")+
  scale_x_discrete(limits = c(seq(0.5,5,0.5))) +
  labs(title = "Chart 1 Ratings by Scoring Scheme",
  subtitle = "edx dataset", x = "Ratings" )+
  theme_classic()

# Show ratings tabulated by score
table(edx$rating)

# Summarize rating scores 
summary(edx$rating)

# Add date feature to examine trend quality
edx <- mutate(edx, date = as_datetime(timestamp))
class(edx$date) 

edx %>% mutate(date = round_date(date, unit = "week")) %>% 
  group_by(date) %>%
  summarize(rating = mean(rating)) %>% 
  ggplot(aes(date,  rating))  + 
  geom_point() +
  geom_smooth()+
  labs(title="Chart 2 Movie Ratings Trend", 
  subtitle = "edx dataset", x = "Year") +
  theme_classic() 

# Show Chart 3 movies rated by userId
edx %>% group_by(userId) %>% summarize(n = n()) %>%
  ggplot()+
  geom_histogram(aes(n), bins = 30, 
  fill = "yellow", color = "blue")+
  scale_x_log10() +
  labs(title = "Chart 3 Ratings by Users", 
  subtitle = "edx dataset", x="UserId",y="Number")+
  theme_classic()

# Show Chart 4 number ratings by movieId 
edx %>%
  count(movieId) %>%
  ggplot(aes(n)) +
  geom_histogram( bins=30, fill = "yellow", color = "blue") +
  scale_x_log10() +
  labs(title = "Chart 4 Ratings by Movies", subtitle = "edx dataset", 
  x="MovieId", y = "Number")+
  theme_classic()     

# Get feel of simple prediction
mu <- mean(edx$rating)
mu

naive_rmse <- RMSE(edx$rating, mu)
naive_rmse

# Save prediction rmse
rmse_results <- tibble(method = "Average movie rating model", 
RMSE = naive_rmse)
print(rmse_results %>% knitr::kable())

movie_avgs <- edx %>%
  group_by(movieId) %>%
  summarize(b_i = mean(rating - mu))

# Show Chart 5 movie effect
movie_avgs  %>%  
  ggplot(aes(b_i)) +
  geom_histogram(bins = 10, fill="beige", color= "blue")+
  labs(title="Chart 5 Movie Effect", subtitle="edx dataset")+
  theme_classic()    

# Test and save rmse results for validation test   
predicted_ratings <- mu +  edx %>%
  left_join(movie_avgs, by='movieId') %>%
  pull(b_i)

model_1_rmse <- RMSE(predicted_ratings, edx$rating) 

rmse_results <- bind_rows(rmse_results, 
tibble(method="Movie effect model",  
RMSE = model_1_rmse ))

# model_1 results
print(rmse_results %>% knitr::kable())

# Show Chart 6 user effect
edx %>%
  group_by(userId) %>% 
  summarize(b_u = mean(rating)) %>% 
  filter(n()>=100) %>% 
  ggplot(aes(b_u)) +
  geom_histogram(bins = 10, fill="beige", color= "blue")+
  labs(title="Chart 6 User Effect", subtitle="edx dataset")+
  theme_classic()  

user_avgs <- edx %>%             
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

# Summarize 
predicted_ratings <- edx %>%   
  left_join(movie_avgs, by='movieId') %>% 
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

model_2_rmse <- RMSE(predicted_ratings, edx$rating)

rmse_results <- bind_rows(rmse_results,
data.frame(method="Movie and user effect model",
RMSE = model_2_rmse))

# Check result
print(rmse_results %>% knitr::kable())

# tuning lambdas
lambdas <- seq(0, 10, 0.25)

rmses <- sapply(lambdas, function(l){
  
  # edx ratings mean
  mu <- mean(edx$rating)
  
  # adjust mean by movie effect/penalize low ratings edx
  b_i <- edx %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  # adjust mean by user/movie effect/penalize low ratings edx
  b_u <- edx %>%
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  # develop opt penalty lambda validation
  
  # Now using validation
  predicted_ratings <- validation %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  RMSE(predicted_ratings, validation$rating)
})

# Show Chart 7 rmses vs lambdas to select the optimal lambda
qplot(lambdas, rmses, col = I("blue"),
main = "Chart 7 Lambda - RMSE")

# The optimal lambda
lambda <- lambdas[which.min(rmses)]
lambda 

# Test and save results
rmse_results <- bind_rows(rmse_results,
data.frame(method="Regularized movie and user effect model",
RMSE = min(rmses)))

# Print results
rmse_results %>% knitr::kable()