# %GETMOVIELIST reads the fixed movie list in movie.txt and returns a
# %cell array of the words
# %   movieList = GETMOVIELIST() reads the fixed movie list in movie.txt
# %   and returns a cell array of the words in movieList.
#
#
# %% Read the fixed movieulary list
# fid = fopen('movie_ids.txt');
#
# % Store all movies in cell array movie{}
# n = 1682;  % Total number of movies
#
# movieList = cell(n, 1);
# for i = 1:n
#     % Read line
#     line = fgets(fid);
#     % Word Index (can ignore since it will be = i)
#     [idx, movieName] = strtok(line, ' ');
#     % Actual Word
#     movieList{i} = strtrim(movieName);
# end
# fclose(fid);
#
# end


def loadMovieList():
    with open('movie_ids.txt', 'r', encoding='ISO-8859-1') as movies:
        lines = movies.readlines()
        movie_titles = [line.strip().split(' ', 1)[1] for line in lines]

    return movie_titles