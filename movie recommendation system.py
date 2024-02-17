#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# **preprocessing:**

# In[4]:


movies_df = pd.read_csv('C:/Users/MAUSAM/Desktop/movies.csv')
ratings_df = pd.read_csv('C:/Users/MAUSAM/Desktop/ratings.csv')


# In[5]:


movies_df.head()


# In[6]:


movies_df['year'] = movies_df.title.str.extract('(\(\d\d\d\d\))',expand=False)
movies_df['year'] = movies_df.year.str.extract('(\d\d\d\d)',expand=False)       #removing parenthesis
movies_df['title'] = movies_df.title.str.replace('(\(\d\d\d\d\))', '')          #removing years from the title column
movies_df['title'] = movies_df['title'].apply(lambda x: x.strip())              #removing white spaces


# In[7]:


#results:
movies_df.head()


# In[8]:


# Final movie dataframe:
ratings_df.head()


# now,Every row in the ratings dataframe has a user id associated with at least one movie, a rating and a timestamp showing when they reviewed it.

# # EDA

# # How many movies were released each year

# In[9]:


# Group movies by year and count the number of movies
movie_counts = movies_df['year'].value_counts().sort_index()

# Plotting the number of movies released each year
plt.figure(figsize=(10, 6))
plt.bar(movie_counts.index, movie_counts.values)
plt.xlabel('Year')
plt.ylabel('Number of Movies')
plt.title('Number of Movies Released Each Year')
plt.show()


# # Difference of Years Between Release Date of the Movie and the Date Rating Was Given

# In[10]:


#merging both the data sets:
merged_data = movies_df.merge(ratings_df, on='movieId')
merged_data.head()


# In[11]:


#Calculating the year difference:-
merged_data['release_date'] = pd.to_datetime(merged_data['year'], format='%Y')
merged_data['timestamp'] = pd.to_datetime(merged_data['timestamp'], unit='s')   #converting into datetime data type

merged_data['year_difference'] = (merged_data['timestamp'].dt.year - merged_data['release_date'].dt.year) #year difference between release date and rating date.

print(merged_data[['movieId', 'title', 'release_date', 'timestamp', 'year_difference']])    #Year difference for each moving rate.


# In[12]:


# Plotting the year difference distribution
plt.figure(figsize=(10, 6))
plt.hist(merged_data['year_difference'], bins=15, edgecolor='black')
plt.xlabel('Year Difference')
plt.ylabel('Frequency')
plt.title('Distribution of Year Difference between Release Date and Rating Date')
plt.show()


# # The Top 10 rated movies

# In[13]:


movie_ratings = merged_data['title'].value_counts().head(10)     #number of ratings for each movie

plt.figure(figsize=(10, 6))                                      #plotting the top 10 rated movies.
sns.barplot(x=movie_ratings.values, y=movie_ratings.index)
plt.xlabel('Number of Ratings')
plt.ylabel('Movie Title')
plt.title('Top 10 Most Rated Movies')
plt.show()


# # How are the ratings distributed across all the movies

# In[14]:


plt.figure(figsize=(10, 6))
sns.histplot(ratings_df['rating'], kde=True)
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.title('Distribution of Movie Ratings')
plt.show()


# ## Distribution of ratings for a specific movies 

# In[15]:


movie_id = 1  # Specify the movie ID for analysis

movie_ratings = ratings_df[ratings_df['movieId'] == movie_id]['rating']      #Filter ratings data for the specific movie

# Plotting the distribution of ratings for the specific movie
plt.figure(figsize=(10, 6))
sns.histplot(movie_ratings, kde=True)
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.title(f'Distribution of Ratings for Movie {movie_id}')
plt.show()


# ## Average rating for each movie genre 

# In[16]:


average_rating_by_genre = merged_data.groupby('genres')['rating'].mean()

print(average_rating_by_genre)


# ## Average rating for each year:

# In[17]:


# Calculate the average rating for each year
avg_ratings = merged_data.groupby('year')['rating'].mean()

# Plotting the average rating for each year
plt.figure(figsize=(10, 6))
plt.plot(avg_ratings.index, avg_ratings.values, marker='o')
plt.xlabel('Year')
plt.ylabel('Average Rating')
plt.title('Average Rating for Each Year')
plt.show()


# In[18]:


# Calculate the average rating for each movie
avg_ratings = ratings_df.groupby('movieId')['rating'].mean()
avg_ratings = avg_ratings.sort_values(ascending=False).head(10)

# Get the movie titles for the top 10 highest-rated movies
top_rated_movies = movies_df[movies_df['movieId'].isin(avg_ratings.index)]['title']

# Plotting the top 10 highest-rated movies
plt.figure(figsize=(10, 6))
sns.barplot(x=avg_ratings.values, y=top_rated_movies)
plt.xlabel('Average Rating')
plt.ylabel('Movie Title')
plt.title('Top 10 Highest-Rated Movies')
plt.show()


# ## How many ratings does each user contribute 

# In[19]:


# Count the number of ratings by each user
user_rating_counts = ratings_df['userId'].value_counts()

# Plotting the number of ratings contributed by each user
plt.figure(figsize=(10, 6))
sns.histplot(user_rating_counts, kde=True)
plt.xlabel('Number of Ratings')
plt.ylabel('Frequency')
plt.title('Distribution of User Ratings')
plt.show()


# ## Ratings distributed for a specific genre

# In[20]:


genre = 'Action'  # Specify the genre for analysis

# Filter ratings data for the specific genre
genre_ratings = merged_data[merged_data['genres'] == genre]['rating']

# Plotting the distribution of ratings for the specific genre
plt.figure(figsize=(10, 6))
sns.histplot(genre_ratings, kde=True)
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.title(f'Distribution of Ratings for Genre: {genre}')
plt.show()


# # How are the ratings distributed for a specific movie across different user ratings

# In[24]:


movie_id = 2  # Specify the movie ID for analysis

# Filter ratings data for the specific movie
movie_ratings = merged_data[merged_data['movieId'] == movie_id]

# Plotting the distribution of ratings for the specific movie across different user ratings
plt.figure(figsize=(10, 6))
sns.boxplot(data=movie_ratings, x='userId', y='rating')
plt.xlabel('User ID')
plt.ylabel('Rating')
plt.title(f'Distribution of Ratings for Movie {movie_id} across Different Users')
plt.show()


# # # Collaborative filtering

# In[25]:


userInput = [
            {'title':'Breakfast Club, The', 'rating':5},
            {'title':'Toy Story', 'rating':3.5},
            {'title':'Jumanji', 'rating':2},
            {'title':"Pulp Fiction", 'rating':5},
            {'title':'Akira', 'rating':4.5}
         ] 
inputMovies = pd.DataFrame(userInput)
inputMovies


# In[26]:


# Add movieId to input user\
#Filtering out the movies by title
inputId = movies_df[movies_df['title'].isin(inputMovies['title'].tolist())]
#Then merging it so we can get the movieId. It's implicitly merging it by title.
inputMovies = pd.merge(inputId, inputMovies)
#Dropping information we won't use from the input dataframe
inputMovies = inputMovies.drop('year', 1)
#Final input dataframe
#If a movie you added in above isn't here, then it might not be in the original 
#dataframe or it might spelled differently, please check capitalisation.
inputMovies


# #  The users who has seen the same movies:
# 

# In[27]:


#Filtering out users that have watched movies that the input has watched and storing it
userSubset = ratings_df[ratings_df['movieId'].isin(inputMovies['movieId'].tolist())]
userSubset.head()


# In[ ]:




